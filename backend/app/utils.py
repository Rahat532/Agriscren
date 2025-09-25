from fastapi import HTTPException, UploadFile

def validate_image(file: UploadFile):
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    max_size = 5 * 1024 * 1024  # 5MB
    
    if file.content_type not in allowed_types:
        raise HTTPException(400, "Only JPG/PNG images allowed")
    
    # Size check
    file.file.seek(0, 2)
    if file.file.tell() > max_size:
        raise HTTPException(400, "Image too large (max 5MB)")
    file.file.seek(0)
