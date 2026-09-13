from fastapi import APIRouter, HTTPException
from typing import List
from app.db.database import get_all_devices, create_device, delete_device
from app.db.models import DeviceCreate, DeviceResponse

router = APIRouter(prefix="/api/v1/devices", tags=["ESP32 Device Security"])


@router.get("", response_model=List[DeviceResponse])
async def list_devices():
    """List all registered ESP32 devices and connection status."""
    return get_all_devices()


@router.post("", response_model=DeviceResponse)
async def register_device(payload: DeviceCreate):
    """Generate a new ESP32 device API authorization key."""
    device = create_device(payload.device_name)
    return device


@router.delete("/{device_id}")
async def remove_device(device_id: int):
    """Revoke an ESP32 device authorization key."""
    delete_device(device_id)
    return {"success": True, "message": f"Device {device_id} revoked."}
