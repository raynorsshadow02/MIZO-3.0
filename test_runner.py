import sys
import traceback

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def run():
    print("Running Mizo 3.0 API & Hardware Pipeline Verification Tests...")
    try:
        from tests.test_api import (
            test_health,
            test_admin_settings,
            test_students_api,
            test_knowledge_base,
            test_esp32_chat,
            test_esp32_audio_pipeline
        )

        test_health()
        print("✓ test_health passed")

        test_admin_settings()
        print("✓ test_admin_settings passed")

        test_students_api()
        print("✓ test_students_api passed")

        test_knowledge_base()
        print("✓ test_knowledge_base passed")

        test_esp32_chat()
        print("✓ test_esp32_chat passed")

        test_esp32_audio_pipeline()
        print("✓ test_esp32_audio_pipeline passed")

        print("\n🎉 ALL 6 VERIFICATION TESTS PASSED SUCCESSFULLY!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run())
