//! MCP Transport Layer (deprecated, use rmcp SDK)
//!
//! This module no longer provides a trait-based unified transport interface.
//! Instead, use the official rmcp SDK directly to construct the appropriate
//! transport and client for each protocol (Streamable HTTP, SSE, TCP, Stdio, etc).
//!
//! Example usage for Streamable HTTP transport (see rmcp documentation):
//!
//! ```rust,no_run
//! use anyhow::Result;
//! use rmcp::{
//!     ServiceExt,
//!     model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation},
//!     transport::StreamableHttpClientTransport,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let transport = StreamableHttpClientTransport::from_uri("http://localhost:8000/mcp");
//!     let client_info = ClientInfo {
//!         protocol_version: Default::default(),
//!         capabilities: ClientCapabilities::default(),
//!         client_info: Implementation {
//!             name: "my client".to_string(),
//!             version: "0.0.1".to_string(),
//!         },
//!     };
//!     let client = client_info.serve(transport).await?;
//!     let tools = client.list_tools(Default::default()).await?;
//!     let tool_result = client
//!         .call_tool(CallToolRequestParam {
//!             name: "increment".into(),
//!             arguments: serde_json::json!({}).as_object().cloned(),
//!         })
//!         .await?;
//!     client.cancel().await?;
//!     Ok(())
//! }
//! ```
