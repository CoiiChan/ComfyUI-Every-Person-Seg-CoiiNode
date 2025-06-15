NODE_CLASS_MAPPINGS = {}

try:
    from .every_person_seg_detail import EveryPersonSegDetail
    NODE_CLASS_MAPPINGS["EveryPersonSegDetail"] = EveryPersonSegDetail
except Exception as e:
    print(f"Error loading EveryPersonSegDetail: {e}")
try:
    from .every_person_seg import EveryPersonSegSimple
    NODE_CLASS_MAPPINGS["EveryPersonSegSimple"] = EveryPersonSegSimple
except Exception as e:
    print(f"Error loading EveryPersonSegSimple: {e}")

except Exception as e:
    print(f"Error loading EveryPersonSegYOLOv8: {e}")

NODE_DISPLAY_NAME_MAPPINGS = {

    "EveryPersonSegDetail": "Every Person Seg Detail",
    "EveryPersonSegSimple": "Every Person Seg Simple",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

