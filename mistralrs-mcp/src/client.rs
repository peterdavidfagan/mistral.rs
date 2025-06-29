use rmcp::{
    ServiceExt,
    model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation, Tool as McpSdkTool},
    transport::{StreamableHttpClientTransport, SseClientTransport},
};
use rmcp::transport::common::client_side_sse::ExponentialBackoff;

use crate::tools::{Function, Tool as LocalTool, ToolCallback, ToolCallbackWithTool, ToolType};
use crate::transport::{HttpTransport, McpTransport, ProcessTransport, WebSocketTransport};
use crate::types::McpToolResult;
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
            McpServerSource::StreamableHttp {
                uri,
                max_times,
                base_duration,
                channel_buffer_capacity,
                allow_stateless,
            } => {
                let connection = StreamableHttpMcpConnection::new(
                    config.id.clone(),
                    config.name.clone(),
                    uri.clone(),
                    *max_times,
                    *base_duration,
                    *channel_buffer_capacity,
                    *allow_stateless,
                ).await?;
                Ok(Arc::new(connection))
            }
            McpServerSource::Sse {
                uri,
                max_times,
                base_duration,
                use_message_endpoint,
            } => {
                let connection = SseMcpConnection::new(
                    config.id.clone(),
                    config.name.clone(),
                    uri.clone(),
                    *max_times,
                    *base_duration,
                    use_message_endpoint.clone(),
                ).await?;
                Ok(Arc::new(connection))
            }
            _ => {
                anyhow::bail!("Unsupported MCP server source or legacy code not ported")
            }
            McpServerSource::Http {
                url,
                timeout_secs,
                headers,
            } => {
                // Merge Bearer token with existing headers if provided
                let mut merged_headers = headers.clone().unwrap_or_default();
                if let Some(token) = &config.bearer_token {
                    merged_headers.insert("Authorization".to_string(), format!("Bearer {token}"));
                }

                let connection = HttpMcpConnection::new(
                    config.id.clone(),
                    config.name.clone(),
                    url.clone(),
                    *timeout_secs,
                    Some(merged_headers),
                )
                .await?;
                Ok(Arc::new(connection))
            }
            McpServerSource::Process {
                command,
                args,
                work_dir,
                env,
            } => {
                let connection = ProcessMcpConnection::new(
                    config.id.clone(),
                    config.name.clone(),
                    command.clone(),
                    args.clone(),
                    work_dir.clone(),
                    env.clone(),
                )
                .await?;
                Ok(Arc::new(connection))
            }
            McpServerSource::WebSocket {
                url,
                timeout_secs,
                headers,
            } => {
                // Merge Bearer token with existing headers if provided
                let mut merged_headers = headers.clone().unwrap_or_default();
                if let Some(token) = &config.bearer_token {
                    merged_headers.insert("Authorization".to_string(), format!("Bearer {token}"));
                }

                let connection = WebSocketMcpConnection::new(
                    config.id.clone(),
                    config.name.clone(),
                    url.clone(),
                    *timeout_secs,
                    Some(merged_headers),
                )
                .await?;
                Ok(Arc::new(connection))
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
    pub async fn new(
        server_id: String, 
        server_name: String, 
        uri: String,
        max_times: Option<usize>,
        base_duration: Option<Duration>,
        channel_buffer_capacity: Option<usize>,
        allow_stateless: Option<bool>,
    ) -> Result<Self> {
        let retry_policy = ExponentialBackoff {
            max_times: Some(max_times.unwrap_or(3)),
            base_duration: base_duration.unwrap_or_else(|| Duration::from_millis(100)),
        };
        let config = rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig {
            uri: Arc::<str>::from(uri.as_str()),
            retry_config: Arc::new(retry_policy),
            channel_buffer_capacity: channel_buffer_capacity.unwrap_or(100),
            allow_stateless: allow_stateless.unwrap_or(true),
        };
        let transport = StreamableHttpClientTransport::with_client(reqwest::Client::default(), config);
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

/// SSE MCP server connection using rmcp SDK
pub struct SseMcpConnection {
    client: Arc<rmcp::service::RunningService<rmcp::RoleClient, rmcp::model::InitializeRequestParam>>,
    server_id: String,
    server_name: String,
}

impl SseMcpConnection {
    /// Create a new SSE MCP connection.
    ///
    /// The configuration options max_times, base_duration, and use_message_endpoint are parsed and
    /// logged for future enhanced SSE support.
    pub async fn new(
        server_id: String,
        server_name: String,
        uri: String,
        max_times: Option<usize>,
        base_duration: Option<std::time::Duration>,
        use_message_endpoint: Option<String>,
    ) -> Result<Self> {
        tracing::info!(
            ?max_times,
            ?base_duration,
            ?use_message_endpoint,
            "SSE connection configuration"
        );
        let retry_policy = ExponentialBackoff {
            max_times: Some(max_times.unwrap_or(3)),
            base_duration: base_duration.unwrap_or_else(|| Duration::from_millis(100)),
        };
        let config = rmcp::transport::sse_client::SseClientConfig {
            sse_endpoint: Arc::<str>::from(uri.as_str()),
            retry_policy: Arc::new(retry_policy),
            use_message_endpoint: use_message_endpoint
        };
        // TODO: Integrate max_times, base_duration, and use_message_endpoint into SSE transport when supported.
        let transport = rmcp::transport::SseClientTransport::start_with_client(reqwest::Client::default(), config).await?;
        let client_info = ClientInfo {
            protocol_version: Default::default(),
            capabilities: ClientCapabilities::default(),
            client_info: Implementation {
                name: server_name.clone(),
                version: "0.0.1".to_string(),
            },
        };
        let client: Arc<rmcp::service::RunningService<rmcp::RoleClient, rmcp::model::InitializeRequestParam>> =
            Arc::new(client_info.serve(transport).await?);

        Ok(Self {
            client,
            server_id,
            server_name,
        })
    }
}

#[async_trait::async_trait]
impl McpServerConnection for SseMcpConnection {
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
                annotations: t.annotations.as_ref().map(|a| serde_json::to_value(a).unwrap_or(Value::Null)),
            })
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

/// HTTP-based MCP server connection
pub struct HttpMcpConnection {
    server_id: String,
    server_name: String,
    transport: Arc<dyn McpTransport>,
}

impl HttpMcpConnection {
    pub async fn new(
        server_id: String,
        server_name: String,
        url: String,
        timeout_secs: Option<u64>,
        headers: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let transport = HttpTransport::new(url, timeout_secs, headers)?;

        let connection = Self {
            server_id,
            server_name,
            transport: Arc::new(transport),
        };

        // Initialize the connection
        connection.initialize().await?;

        Ok(connection)
    }

    async fn initialize(&self) -> Result<()> {
        let init_params = serde_json::json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "mistral.rs",
                "version": "0.6.0"
            }
        });

        self.transport
            .send_request("initialize", init_params)
            .await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl McpServerConnection for HttpMcpConnection {
    fn server_id(&self) -> &str {
        &self.server_id
    }

    fn server_name(&self) -> &str {
        &self.server_name
    }

    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let result = self
            .transport
            .send_request("tools/list", Value::Null)
            .await?;

        let tools = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid tools response format"))?;

        let mut tool_infos = Vec::new();
        for tool in tools {
            let name = tool
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| anyhow::anyhow!("Tool missing name"))?
                .to_string();

            let description = tool
                .get("description")
                .and_then(|d| d.as_str())
                .map(|s| s.to_string());

            let input_schema = tool
                .get("inputSchema")
                .cloned()
                .unwrap_or(Value::Object(serde_json::Map::new()));

            tool_infos.push(McpToolInfo {
                name,
                description,
                input_schema,
                server_id: self.server_id.clone(),
                server_name: self.server_name.clone(),
                annotations: None, // Annotations not supported in this implementation
            });
        }

        Ok(tool_infos)
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> Result<String> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let result = self.transport.send_request("tools/call", params).await?;

        // Parse the MCP tool result
        let tool_result: McpToolResult = serde_json::from_value(result)?;

        // Check if the result indicates an error
        if tool_result.is_error.unwrap_or(false) {
            return Err(anyhow::anyhow!(
                "Tool execution failed: {}",
                tool_result.to_string()
            ));
        }

        Ok(tool_result.to_string())
    }

    async fn list_resources(&self) -> Result<Vec<Resource>> {
        let result = self
            .transport
            .send_request("resources/list", Value::Null)
            .await?;

        let resources = result
            .get("resources")
            .and_then(|r| r.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid resources response format"))?;

        let mut resource_list = Vec::new();
        for resource in resources {
            let mcp_resource: Resource = serde_json::from_value(resource.clone())?;
            resource_list.push(mcp_resource);
        }

        Ok(resource_list)
    }

    async fn read_resource(&self, uri: &str) -> Result<String> {
        let params = serde_json::json!({ "uri": uri });
        let result = self
            .transport
            .send_request("resources/read", params)
            .await?;

        // Extract content from the response
        if let Some(contents) = result.get("contents").and_then(|c| c.as_array()) {
            if let Some(first_content) = contents.first() {
                if let Some(text) = first_content.get("text").and_then(|t| t.as_str()) {
                    return Ok(text.to_string());
                }
            }
        }

        Err(anyhow::anyhow!("No readable content found in resource"))
    }

    async fn ping(&self) -> Result<()> {
        // Send a simple ping to check if the server is responsive
        self.transport.send_request("ping", Value::Null).await?;
        Ok(())
    }
}

/// Process-based MCP server connection
pub struct ProcessMcpConnection {
    server_id: String,
    server_name: String,
    transport: Arc<dyn McpTransport>,
}

impl ProcessMcpConnection {
    pub async fn new(
        server_id: String,
        server_name: String,
        command: String,
        args: Vec<String>,
        work_dir: Option<String>,
        env: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let transport = ProcessTransport::new(command, args, work_dir, env).await?;

        let connection = Self {
            server_id,
            server_name,
            transport: Arc::new(transport),
        };

        // Initialize the connection
        connection.initialize().await?;

        Ok(connection)
    }

    async fn initialize(&self) -> Result<()> {
        let init_params = serde_json::json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "mistral.rs",
                "version": "0.6.0"
            }
        });

        self.transport
            .send_request("initialize", init_params)
            .await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl McpServerConnection for ProcessMcpConnection {
    fn server_id(&self) -> &str {
        &self.server_id
    }

    fn server_name(&self) -> &str {
        &self.server_name
    }

    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let result = self
            .transport
            .send_request("tools/list", Value::Null)
            .await?;

        let tools = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid tools response format"))?;

        let mut tool_infos = Vec::new();
        for tool in tools {
            let name = tool
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| anyhow::anyhow!("Tool missing name"))?
                .to_string();

            let description = tool
                .get("description")
                .and_then(|d| d.as_str())
                .map(|s| s.to_string());

            let input_schema = tool
                .get("inputSchema")
                .cloned()
                .unwrap_or(Value::Object(serde_json::Map::new()));

            tool_infos.push(McpToolInfo {
                name,
                description,
                input_schema,
                server_id: self.server_id.clone(),
                server_name: self.server_name.clone(),
                annotations: None, // Annotations not supported in this implementation
            });
        }

        Ok(tool_infos)
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> Result<String> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let result = self.transport.send_request("tools/call", params).await?;

        // Parse the MCP tool result
        let tool_result: McpToolResult = serde_json::from_value(result)?;

        // Check if the result indicates an error
        if tool_result.is_error.unwrap_or(false) {
            return Err(anyhow::anyhow!(
                "Tool execution failed: {}",
                tool_result.to_string()
            ));
        }

        Ok(tool_result.to_string())
    }

    async fn list_resources(&self) -> Result<Vec<Resource>> {
        let result = self
            .transport
            .send_request("resources/list", Value::Null)
            .await?;

        let resources = result
            .get("resources")
            .and_then(|r| r.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid resources response format"))?;

        let mut resource_list = Vec::new();
        for resource in resources {
            let mcp_resource: Resource = serde_json::from_value(resource.clone())?;
            resource_list.push(mcp_resource);
        }

        Ok(resource_list)
    }

    async fn read_resource(&self, uri: &str) -> Result<String> {
        let params = serde_json::json!({ "uri": uri });
        let result = self
            .transport
            .send_request("resources/read", params)
            .await?;

        // Extract content from the response
        if let Some(contents) = result.get("contents").and_then(|c| c.as_array()) {
            if let Some(first_content) = contents.first() {
                if let Some(text) = first_content.get("text").and_then(|t| t.as_str()) {
                    return Ok(text.to_string());
                }
            }
        }

        Err(anyhow::anyhow!("No readable content found in resource"))
    }

    async fn ping(&self) -> Result<()> {
        // Send a simple ping to check if the server is responsive
        self.transport.send_request("ping", Value::Null).await?;
        Ok(())
    }
}

/// WebSocket-based MCP server connection
pub struct WebSocketMcpConnection {
    server_id: String,
    server_name: String,
    transport: Arc<dyn McpTransport>,
}

impl WebSocketMcpConnection {
    pub async fn new(
        server_id: String,
        server_name: String,
        url: String,
        timeout_secs: Option<u64>,
        headers: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let transport = WebSocketTransport::new(url, timeout_secs, headers).await?;

        let connection = Self {
            server_id,
            server_name,
            transport: Arc::new(transport),
        };

        // Initialize the connection
        connection.initialize().await?;

        Ok(connection)
    }

    async fn initialize(&self) -> Result<()> {
        let init_params = serde_json::json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "mistral.rs",
                "version": "0.6.0"
            }
        });

        self.transport
            .send_request("initialize", init_params)
            .await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl McpServerConnection for WebSocketMcpConnection {
    fn server_id(&self) -> &str {
        &self.server_id
    }

    fn server_name(&self) -> &str {
        &self.server_name
    }

    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let result = self
            .transport
            .send_request("tools/list", Value::Null)
            .await?;

        let tools = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid tools response format"))?;

        let mut tool_infos = Vec::new();
        for tool in tools {
            let name = tool
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| anyhow::anyhow!("Tool missing name"))?
                .to_string();

            let description = tool
                .get("description")
                .and_then(|d| d.as_str())
                .map(|s| s.to_string());

            let input_schema = tool
                .get("inputSchema")
                .cloned()
                .unwrap_or(Value::Object(serde_json::Map::new()));

            tool_infos.push(McpToolInfo {
                name,
                description,
                input_schema,
                server_id: self.server_id.clone(),
                server_name: self.server_name.clone(),
                annotations: None, // Annotations not supported in this implementation
            });
        }

        Ok(tool_infos)
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> Result<String> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let result = self.transport.send_request("tools/call", params).await?;

        // Parse the MCP tool result
        let tool_result: McpToolResult = serde_json::from_value(result)?;

        // Check if the result indicates an error
        if tool_result.is_error.unwrap_or(false) {
            return Err(anyhow::anyhow!(
                "Tool execution failed: {}",
                tool_result.to_string()
            ));
        }

        Ok(tool_result.to_string())
    }

    async fn list_resources(&self) -> Result<Vec<Resource>> {
        let result = self
            .transport
            .send_request("resources/list", Value::Null)
            .await?;

        let resources = result
            .get("resources")
            .and_then(|r| r.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid resources response format"))?;

        let mut resource_list = Vec::new();
        for resource in resources {
            let mcp_resource: Resource = serde_json::from_value(resource.clone())?;
            resource_list.push(mcp_resource);
        }

        Ok(resource_list)
    }

    async fn read_resource(&self, uri: &str) -> Result<String> {
        let params = serde_json::json!({ "uri": uri });
        let result = self
            .transport
            .send_request("resources/read", params)
            .await?;

        // Extract content from the response
        if let Some(contents) = result.get("contents").and_then(|c| c.as_array()) {
            if let Some(first_content) = contents.first() {
                if let Some(text) = first_content.get("text").and_then(|t| t.as_str()) {
                    return Ok(text.to_string());
                }
            }
        }

        Err(anyhow::anyhow!("No readable content found in resource"))
    }

    async fn ping(&self) -> Result<()> {
        // Send a simple ping to check if the server is responsive
        self.transport.send_request("ping", Value::Null).await?;
        Ok(())
    }
}