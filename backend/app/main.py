import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import inspect, text

from app.api.routes import (
    action_drafts,
    chat,
    ci,
    code_graph,
    conversations,
    evaluations,
    feedback,
    issues,
    knowledge,
    project_index,
    pull_requests,
    rag,
    repos,
    reports,
    search,
    skills,
    webhooks,
    workspaces,
)
from app.core.config import settings
from app.db.models import Base
from app.db.session import engine
from app.services.scheduler import periodic_repo_sync_loop
from app.services.rag.vector_store import MilvusUnavailableError


def _cors_origins() -> list[str]:
    """把配置中的逗号分隔来源转换成 CORS 中间件需要的列表。"""
    return [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]


def _cors_origin_regex() -> str | None:
    """开发环境额外允许本机任意端口；生产环境只使用显式配置的来源。"""
    if settings.app_env.lower() == "production":
        return None
    return r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"


def create_app() -> FastAPI:
    """集中注册 API、错误处理和应用生命周期任务。"""
    app = FastAPI(title=settings.app_name, version="0.1.0")

    # 将向量库故障统一转换为 503，便于客户端区分服务降级与普通请求错误。
    @app.exception_handler(MilvusUnavailableError)
    async def milvus_unavailable_handler(_request, exc: MilvusUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "service": "milvus", "status": "degraded"},
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_origin_regex=_cors_origin_regex(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 最终路径由这里的 prefix 与各路由文件中的路径拼接；tags 用于 OpenAPI 分组。
    app.include_router(repos.router, prefix="/api/repos", tags=["repos"])
    app.include_router(conversations.router, prefix="/api/repos", tags=["conversations"])
    app.include_router(workspaces.router, prefix="/api/workspaces", tags=["workspaces"])
    app.include_router(issues.router, prefix="/api", tags=["issues"])
    app.include_router(pull_requests.router, prefix="/api", tags=["pull_requests"])
    app.include_router(ci.router, prefix="/api", tags=["ci"])
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(feedback.router, prefix="/api/chat/feedback", tags=["chat_feedback"])
    app.include_router(action_drafts.router, prefix="/api/action-drafts", tags=["action_drafts"])
    app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
    app.include_router(search.router, prefix="/api/search", tags=["search"])
    app.include_router(skills.router, prefix="/api/skills", tags=["skills"])
    app.include_router(knowledge.router, prefix="/api/repos", tags=["knowledge"])
    app.include_router(project_index.router, prefix="/api/repos", tags=["project_index"])
    app.include_router(rag.router, prefix="/api/rag", tags=["rag"])
    app.include_router(code_graph.router, prefix="/api", tags=["code_graph"])
    app.include_router(evaluations.router, prefix="/api/evals", tags=["evaluations"])
    app.include_router(webhooks.router, prefix="/api/webhooks", tags=["webhooks"])
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "app": settings.app_name}

    @app.on_event("startup")
    async def create_demo_tables() -> None:
        # 课程项目默认自动建表；生产风格流程会在后续加入 Alembic 迁移。
        Base.metadata.create_all(bind=engine)
        # create_all 不会修改已有表，因此下面按列是否存在来兼容旧数据库。
        inspector = inspect(engine)
        if "users" in inspector.get_table_names():
            columns = {column["name"] for column in inspector.get_columns("users")}
            if "role" not in columns:
                with engine.begin() as connection:
                    connection.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR(80) DEFAULT 'owner'"))
        if "repositories" in inspector.get_table_names():
            columns = {column["name"] for column in inspector.get_columns("repositories")}
            with engine.begin() as connection:
                if "provider" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN provider VARCHAR(80) DEFAULT 'github'"))
                if "api_base_url" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN api_base_url TEXT"))
                if "clone_url" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN clone_url TEXT"))
                if "local_path" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN local_path TEXT"))
                if "checkout_mode" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN checkout_mode VARCHAR(80) DEFAULT 'managed'"))
                if "github_token_encrypted" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN github_token_encrypted TEXT"))
        if "workflow_runs" in inspector.get_table_names():
            columns = {column["name"] for column in inspector.get_columns("workflow_runs")}
            if "jobs" not in columns:
                with engine.begin() as connection:
                    connection.execute(text("ALTER TABLE workflow_runs ADD COLUMN jobs JSON"))
        for table_name in ["code_symbols", "code_relations"]:
            if table_name in inspector.get_table_names():
                columns = {column["name"] for column in inspector.get_columns(table_name)}
                with engine.begin() as connection:
                    if "pr_id" not in columns:
                        connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN pr_id UUID"))
                    if "pr_number" not in columns:
                        connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN pr_number INTEGER"))
        for table_name in ["chat_messages", "chat_sessions", "agent_runs", "evidence_items"]:
            if table_name in inspector.get_table_names():
                columns = {column["name"] for column in inspector.get_columns(table_name)}
                if "conversation_id" not in columns:
                    with engine.begin() as connection:
                        connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN conversation_id UUID"))
        for table_name in ["thread_memories", "conversation_memories"]:
            if table_name in inspector.get_table_names():
                columns = {column["name"] for column in inspector.get_columns(table_name)}
                for column_name in ["facts", "tasks", "user_preferences", "repo_context", "citations"]:
                    if column_name not in columns:
                        with engine.begin() as connection:
                            connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} JSON"))
        if "documents" in inspector.get_table_names():
            document_columns = {column["name"] for column in inspector.get_columns("documents")}
            if "embedding" in document_columns:
                # 向量数据已移至 Milvus，清理旧表中不再使用的 embedding 列。
                with engine.begin() as connection:
                    connection.execute(text("ALTER TABLE documents DROP COLUMN embedding"))

        # 这些导入只在启动时的数据修复阶段使用。
        from app.db.session import SessionLocal
        from app.services.conversations import backfill_all_default_conversations
        from app.services.code_analysis import purge_legacy_code_documents
        from app.services.rag.vector_store import delete_milvus_documents, milvus_runtime_status

        db = SessionLocal()
        try:
            # 为已有仓库补默认会话，并移除旧的代码文件文档记录。
            backfill_all_default_conversations(db)
            if purge_legacy_code_documents(db):
                db.commit()
        finally:
            db.close()
        # 记录向量库状态；清理旧代码向量时允许 Milvus 暂时不可用。
        app.state.rag_vector_status = milvus_runtime_status()
        delete_milvus_documents(source_types=["code_file"], strict=False)
        if settings.auto_sync_enabled:
            # 后台循环定期同步仓库，不阻塞启动流程。
            app.state.repo_sync_task = asyncio.create_task(periodic_repo_sync_loop())

    @app.on_event("shutdown")
    async def stop_background_tasks() -> None:
        task = getattr(app.state, "repo_sync_task", None)
        if task:
            # 关闭应用时通知同步循环取消，避免留下后台任务。
            task.cancel()

    return app


# uvicorn app.main:app 加载的就是这个应用实例。
app = create_app()
