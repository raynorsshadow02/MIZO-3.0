import sys
import uvicorn
from app.config import settings

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    print("=" * 60)
    print(">> MIZO 3.0 CLOUD SERVER STARTING")
    print(f">> Serving at: http://localhost:{settings.PORT}")
    print(f">> Admin Dashboard: http://localhost:{settings.PORT}/")
    print(f">> ESP32 Endpoint: http://localhost:{settings.PORT}/api/v1/esp32/audio")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
