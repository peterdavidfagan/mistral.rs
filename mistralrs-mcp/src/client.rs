use crate::tools::{Function, ToolCallback, ToolCallbackWithTool, ToolType, Tool as LocalTool};
// Remove legacy transport imports
// Add rmcp SDK imports for Streamable HTTP MCP connection
use rmcp::{
    ServiceExt,
    model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation, Tool as McpSdkTool},
    transport::StreamableHttpClientTransport,
};
use crate::{McpClientConfig, McpServerConfig, McpServerSource, McpToolInfo};
use anyhow::Result;
use rust_mcp_schema::Resource;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

/// Trait for MCP server connections
#[async_trait::async_trait]
pub trait McpServerConnection: Send + Sync {
    /// Get the server ID
    fn server_id(&self) -> &str;

    /// Get the server name
    fn server_name(&self) -> &str;

    /// List available tools from this server
    async fn list_tools(&self) -> Result<Vec<McpToolInfo>>;

    /// Call a tool on this server
    async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<String>;

    /// List available resources from this server
    async fn list_resources(&self) -> Result<Vec<Resource>>;

    /// Read a resource from this server
    async fn read_resource(&self, uri: &str) -> Result<String>;

    /// Check if the connection is healthy
    async fn ping(&self) -> Result<()>;
}

/// Converts an McpToolInfo to an internal Tool definition.
fn tool_from_mcp(info: &McpToolInfo) -> LocalTool {
    // Convert input_schema (serde_json::Value) to Option<HashMap<String, Value>>
    let parameters = if let serde_json::Value::Object(ref obj) = info.input_schema {
        // Try "properties" first, fallback to flat object
        if let Some(serde_json::Value::Object(props)) = obj.get("properties") {
            Some(props.clone())
        } else {
            Some(obj.clone())
        }
    } else {
        None
    };

    LocalTool {
        tp: ToolType::Function,
        function: Function {
            name: info.name.clone(),
            description: info.description.clone(),
            parameters: if let serde_json::Value::Object(ref obj) = info.input_schema {
                if let Some(serde_json::Value::Object(props)) = obj.get("properties") {
                    Some(props.clone().into_iter().collect())
                } else {
                    Some(obj.clone().into_iter().collect())
                }
            } else {
                None
            },
        },
    }
}

/// MCP client that manages connections to multiple MCP servers
///
/// The main interface for interacting with Model Context Protocol servers.
/// Handles connection lifecycle, tool discovery, and provides integration
/// with tool calling systems.
///
/// # Features
///
/// - **Multi-server Management**: Connects to and manages multiple MCP servers simultaneously
/// - **Automatic Tool Discovery**: Discovers available tools from connected servers
/// - **Tool Registration**: Converts MCP tools to internal Tool format for seamless integration
/// - **Connection Pooling**: Maintains persistent connections for efficient tool execution
/// - **Error Handling**: Robust error handling with proper cleanup and reconnection logic
///
/// # Example
///
/// ```rust,no_run
/// use mistralrs_mcp::{McpClient, McpClientConfig};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let config = McpClientConfig::default();
///     let mut client = McpClient::new(config);
///     
///     // Initialize all configured server connections
///     client.initialize().await?;
///     
///     // Get tool callbacks for model integration
///     let callbacks = client.get_tool_callbacks_with_tools();
///     
///     Ok(())
/// }
/// ```
pub struct McpClient {
    /// Configuration for the client including server list and policies
    config: McpClientConfig,
    /// Active connections to MCP servers, indexed by server ID
    servers: HashMap<String, Arc<dyn McpServerConnection>>,
    /// Registry of discovered tools from all connected servers
    tools: HashMap<String, McpToolInfo>,
    /// Legacy tool callbacks for backward compatibility
    tool_callbacks: HashMap<String, Arc<ToolCallback>>,
    /// Tool callbacks with associated Tool definitions for automatic tool calling
    tool_callbacks_with_tools: HashMap<String, ToolCallbackWithTool>,
    /// Semaphore to control maximum concurrent tool calls
    concurrency_semaphore: Arc<Semaphore>,
}

impl McpClient {
    /// Create a new MCP client with the given configuration
    pub fn new(config: McpClientConfig) -> Self {
        let max_concurrent = config.max_concurrent_calls.unwrap_or(10);
        Self {
            config,
            servers: HashMap::new(),
            tools: HashMap::new(),
            tool_callbacks: HashMap::new(),
            tool_callbacks_with_tools: HashMap::new(),
            concurrency_semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    /// Initialize connections to all configured servers
    pub async fn initialize(&mut self) -> Result<()> {
        for server_config in &self.config.servers {
            if server_config.enabled {
                let connection = self.create_connection(server_config).await?;
                self.servers.insert(server_config.id.clone(), connection);
            }
        }

        if self.config.auto_register_tools {
            self.discover_and_register_tools().await?;
        }

        Ok(())
    }

    /// Get tool callbacks that can be used with the existing tool calling system
    pub fn get_tool_callbacks(&self) -> &HashMap<String, Arc<ToolCallback>> {
        &self.tool_callbacks
    }

    /// Get tool callbacks with their associated Tool definitions
    pub fn get_tool_callbacks_with_tools(&self) -> &HashMap<String, ToolCallbackWithTool> {
        &self.tool_callbacks_with_tools
    }

    /// Get discovered tools information
    pub fn get_tools(&self) -> &HashMap<String, McpToolInfo> {
        &self.tools
    }

    /// Create connection based on server source type
    async fn create_connection(
        &self,
        config: &McpServerConfig,
    ) -> Result<Arc<dyn McpServerConnection>> {
        match &config.source {
            McpServerSource::StreamableHttp { url } => {
                let connection = StreamableHttpMcpConnection::new(
                    config.id.clone(),
                    config.name.clone(),
                    url.clone(),
                ).await?;
                Ok(Arc::new(connection))
            }
            _ => {
                anyhow::bail!("Unsupported MCP server source or legacy code not ported")
            }
        }
    }

    /// Discover tools from all connected servers and register them
    async fn discover_and_register_tools(&mut self) -> Result<()> {
        for (server_id, connection) in &self.servers {
            let tools = connection.list_tools().await?;
            let server_config = self
                .config
                .servers
                .iter()
                .find(|s| &s.id == server_id)
                .ok_or_else(|| anyhow::anyhow!("Server config not found for {}", server_id))?;

            for tool in tools {
                let tool_name = if let Some(prefix) = &server_config.tool_prefix {
                    format!("{}_{}", prefix, tool.name)
                } else {
                    tool.name.clone()
                };

                // Create tool callback that calls the MCP server with timeout and concurrency controls
                let connection_clone = Arc::clone(connection);
                let original_tool_name = tool.name.clone();
                let semaphore_clone = Arc::clone(&self.concurrency_semaphore);
                let timeout_duration =
                    Duration::from_secs(self.config.tool_timeout_secs.unwrap_or(30));

                let callback: Arc<ToolCallback> = Arc::new(move |called_function| {
                    let connection = Arc::clone(&connection_clone);
                    let tool_name = original_tool_name.clone();
                    let semaphore = Arc::clone(&semaphore_clone);
                    let arguments: serde_json::Value =
                        serde_json::from_str(&called_function.arguments)?;

                    // Use tokio::task::spawn_blocking to handle the async-to-sync bridge
                    let rt = tokio::runtime::Handle::current();
                    std::thread::spawn(move || {
                        rt.block_on(async move {
                            // Acquire semaphore permit for concurrency control
                            let _permit = semaphore.acquire().await.map_err(|_| {
                                anyhow::anyhow!("Failed to acquire concurrency permit")
                            })?;

                            // Execute tool call with timeout
                            match tokio::time::timeout(
                                timeout_duration,
                                connection.call_tool(&tool_name, arguments),
                            )
                            .await
                            {
                                Ok(result) => result,
                                Err(_) => Err(anyhow::anyhow!(
                                    "Tool call timed out after {} seconds",
                                    timeout_duration.as_secs()
                                )),
                            }
                        })
                    })
                    .join()
                    .map_err(|_| anyhow::anyhow!("Tool call thread panicked"))?
                });

                let mut tool_info = tool.clone();
                tool_info.name = tool_name.clone(); // Update name with prefix if present
                let tool_def = tool_from_mcp(&tool_info);

                // Store in both collections for backward compatibility
                self.tool_callbacks
                    .insert(tool_name.clone(), callback.clone());
                self.tool_callbacks_with_tools.insert(
                    tool_name.clone(),
                    ToolCallbackWithTool {
                        callback,
                        tool: tool_def,
                    },
                );
                self.tools.insert(tool_name, tool);
            }
        }

        Ok(())
    }

    /// Convert MCP tool input schema to Tool parameters format
    fn convert_mcp_schema_to_parameters(
        schema: &serde_json::Value,
    ) -> Option<HashMap<String, serde_json::Value>> {
        // MCP tools can have various schema formats, we'll try to convert common ones
        match schema {
            serde_json::Value::Object(obj) => {
                let mut params = HashMap::new();

                // If it's a JSON schema object, extract properties
                if let Some(properties) = obj.get("properties") {
                    if let serde_json::Value::Object(props) = properties {
                        for (key, value) in props {
                            params.insert(key.clone(), value.clone());
                        }
                    }
                } else {
                    // If it's just a direct object, use it as-is
                    for (key, value) in obj {
                        params.insert(key.clone(), value.clone());
                    }
                }

                if params.is_empty() {
                    None
                } else {
                    Some(params)
                }
            }
            _ => {
                // For non-object schemas, we can't easily convert to parameters
                None
            }
        }
    }
}


/// Streamable HTTP MCP server connection using rmcp SDK
pub struct StreamableHttpMcpConnection {
    client: Arc<rmcp::service::RunningService<rmcp::RoleClient, rmcp::model::InitializeRequestParam>>,
    server_id: String,
    server_name: String,
}

impl StreamableHttpMcpConnection {
    pub async fn new(server_id: String, server_name: String, url: String) -> Result<Self> {
        let transport = StreamableHttpClientTransport::from_uri(Arc::<str>::from(url.as_str()));
        let client_info = ClientInfo {
            protocol_version: Default::default(),
            capabilities: ClientCapabilities::default(),
            client_info: Implementation {
                name: server_name.clone(),
                version: "0.0.1".to_string(),
            },
        };
        let client: Arc<rmcp::service::RunningService<rmcp::RoleClient, rmcp::model::InitializeRequestParam>> = Arc::new(client_info.serve(transport).await?);

        Ok(Self {
            client,
            server_id,
            server_name,
        })
    }
}

#[async_trait::async_trait]
impl McpServerConnection for StreamableHttpMcpConnection {
    fn server_id(&self) -> &str {
        &self.server_id
    }
    fn server_name(&self) -> &str {
        &self.server_name
    }
    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let result = self.client.list_tools(Default::default()).await?;
        let tools: Vec<McpSdkTool> = result.tools;
        Ok(tools
            .into_iter()
           .map(|t| McpToolInfo {
                name: t.name.to_string(),
                description: t.description.as_ref().map(|c| c.to_string()),
                input_schema: serde_json::Value::Object((*t.input_schema).clone()),
                server_id: self.server_id.clone(),
                server_name: self.server_name.clone(),
                annotations: t.annotations.as_ref().map(|a| serde_json::to_value(a).unwrap_or(Value::Null)),            })
            .collect())
    }
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<String> {
        let result = self
            .client
            .call_tool(CallToolRequestParam {
                name: name.to_string().into(),
                arguments: arguments.as_object().cloned(),
            })
            .await?;
        Ok(format!("{result:?}"))
    }
    async fn list_resources(&self) -> Result<Vec<Resource>> {
        Ok(vec![])
    }
    async fn read_resource(&self, _uri: &str) -> Result<String> {
        Ok(String::new())
    }
    async fn ping(&self) -> Result<()> {
        Ok(())
    }
}
