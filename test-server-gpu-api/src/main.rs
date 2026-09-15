use tokio::runtime::Runtime;
use axum::{routing::get, Router, response::Html};
use tower_http::services::ServeDir;

fn main() {
    env_logger::init();

    let rt = Runtime::new().expect("Failed to create runtime");
    rt.spawn(start(12345));    
}

pub async fn start(port: u16) {
    let app = Router::new()
        .route("/app/test", get(|| async { index("test", "test-app") }))
        .nest_service("/app/test/web", ServeDir::new("../web"));
    
    let addr = "localhost:".to_owned() +& port.to_string();
    
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("Failed to bind tcp listener");

    axum::serve(listener, app).await.expect("Failed to start server");
}

fn index(app_name: &str, app_file_name: &str) -> Html<String> {
	let prefix = "/app/".to_owned() + app_name;        
    let js_path = prefix.clone() + "/web/" + app_file_name + ".js";    
     
    Html(
	    format!(r#"
<!doctype html>    
    <head>
        <meta charset="utf-8" />
        <title>kosmo-ui</title>
    </head>

    <body> 
        <canvas id="canvas"></canvas>
        <script type="module">
            import init from "{}";

            await init();
        </script>
    </body>
</html>
"#, js_path)
    )
}
