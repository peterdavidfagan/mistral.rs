use async_trait::async_trait;
use mistralrs_core::Tool;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use std::future::Future;
use tokio::net::TcpListener;

use rmcp::{
    Error as McpError, RoleServer, ServerHandler,
    handler::server::{router::tool::ToolRouter, tool::Parameters},
    model::*,
    schemars,
    service::RequestContext,
    tool, tool_handler, tool_router,
};

use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};

use mistralrs_server_core::{
    chat_completion::parse_request, handler_core::create_response_channel,
    types::SharedMistralRsState,
};

use mistralrs_server_core::openai::ChatCompletionRequest;

#[derive(Clone, Serialize, Deserialize, Debug, rmcp::schemars::JsonSchema)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

const MCP_INSTRUCTIONS: &str = r#"
This server provides LLM text and multimodal model inference. You can use the following tools:
- `chat` for sending a chat completion request with a model message history
"#;

#[derive(Clone)]
pub struct MistralRsServer {
    state: Arc<SharedMistralRsState>,
    tool_router: ToolRouter<MistralRsServer>,
}

#[rmcp::tool_router]
impl MistralRsServer {
    #[allow(dead_code)]
    pub fn new(state: Arc<SharedMistralRsState>) -> Self {  
        Self { 
            state,
            tool_router: Self::tool_router(),
        }
    }

    #[rmcp::tool(name = "chat", description = "Send a chat completion request with messages and other hyperparameters.")]
    async fn chat_completion(
        &self,
        Parameters(chat_req): Parameters<ChatCompletionRequest>,
    ) -> Result<CallToolResult, rmcp::Error> {
        let (tx, mut rx) = create_response_channel(None);
        let (request, _is_streaming) = parse_request(chat_req, self.state.clone(), tx)
            .await
            .map_err(|e| rmcp::Error::internal(e.to_string()))?;

        mistralrs_server_core::handler_core::send_request(&self.state, request)
            .await
            .map_err(|e| rmcp::Error::internal(e.to_string()))?;

        match rx.recv().await {
            Some(mistralrs_core::Response::Done(resp)) => {
                let content = resp
                    .choices
                    .iter()
                    .filter_map(|c| c.message.content.clone())
                    .collect::<Vec<_>>()
                    .join("\n");

                Ok(CallToolResult {
                    content: vec![CallToolResultContentItem::TextContent(TextContent::new(
                        content, None,
                    ))],
                    is_error: None,
                })
            }
            Some(mistralrs_core::Response::ModelError(msg, _)) => {
                Err(McpError(msg))
            }
            Some(_) | None => Err(McpError("no response".to_string())),
        }
    }
}

#[rmcp::tool_handler]
impl rmcp::ServerHandler for MistralRsServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_prompts()
                .enable_resources()
                .enable_tools()
                .build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(MCP_INSTRUCTIONS.to_string()),
        }
    }

    async fn initialize(
        &self,
        _request: InitializeRequestParam,
        context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, McpError> {
        if let Some(http_request_part) = context.extensions.get::<axum::http::request::Parts>() {
            let initialize_headers = &http_request_part.headers;
            let initialize_uri = &http_request_part.uri;
        }
        Ok(self.get_info())
    }
}

pub async fn create_http_mcp_server(
    state: SharedMistralRsState,
    host: String,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = format!("{host}:{port}").parse()?;

    let state = Arc::new(state);
    let service = StreamableHttpService::new( 
        move || Ok(MistralRsServer::new(state.clone())),
        LocalSessionManager::default().into(),
        Default::default(),
    );

    let app = axum::Router::new().nest_service("/mcp", service);
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
