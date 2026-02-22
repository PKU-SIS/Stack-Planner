"""
Server script for running the DeerFlow API.
"""

import argparse
from src.utils.logger import logger
import signal
import sys
import uvicorn


def enable_debug_logging():
    """Enable debug level logging."""
    logger.set_log_level(log_level="DEBUG")


def handle_shutdown(signum, frame):
    """Handle graceful shutdown on SIGTERM/SIGINT"""
    logger.info("Received shutdown signal. Starting graceful shutdown...")
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the DeerFlow API server")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (default: True except on Windows)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the server to (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    # Determine reload setting
    reload = False
    if args.reload:
        reload = True

    if args.log_level.lower() == "debug":
        enable_debug_logging()
    try:
        logger.info(f"Starting DeerFlow API server on {args.host}:{args.port}")
        uvicorn.run(
            "src.server:app",
            host=args.host,
            port=args.port,
            reload=reload,
            log_level=args.log_level,
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)
