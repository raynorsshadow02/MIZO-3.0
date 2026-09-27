import sys
import pytest

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def run():
    print("=" * 65)
    print(">> RUNNING MIZO 3.0 COMPLETE SYSTEM VERIFICATION SUITE")
    print("=" * 65)
    exit_code = pytest.main(["-v", "tests"])
    if exit_code == 0:
        print("\n🎉 ALL MIZO 3.0 SYSTEM TESTS PASSED SUCCESSFULLY!")
    else:
        print(f"\n❌ Test run failed with code {exit_code}")
    return exit_code

if __name__ == "__main__":
    sys.exit(run())

