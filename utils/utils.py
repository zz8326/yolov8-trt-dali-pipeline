import cv2
import yaml
import random

def load_class_names(config_path="meta_det.yaml"):
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
        return data['names']
    

def draw_detections(image, detections, class_color_map=None):
    """
    image: BGR image (cv2.imread)
    detections: List of detection dicts: [{'bbox': [x0,y0,x1,y1], 'score': float, 'class_id': int or str}]
    class_color_map: Optional dict of class_id -> (B, G, R)
    """
    image = image.copy()
    for det in detections:
        x0, y0, x1, y1 = det['bbox']
        score = det['score']
        cls = det['class_id']
        label = f"{cls}: {score:.2f}"


        color = class_color_map.get(cls, (0, 255, 0)) if class_color_map else (0, 255, 0)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

        ((text_width, text_height), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x0, y0 - text_height - 4), (x0 + text_width, y0), color, -1)
        cv2.putText(image, label, (x0, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image

def get_class_color_map(classes):
    color_map = {}
    for cls in classes:
        color_map[cls] = tuple(random.randint(0, 255) for _ in range(3))
    return color_map



def with_cuda_context(fn):
    def wrapper(self, *args, **kwargs):
        assert hasattr(self, "ctx"), "Missing 'ctx' attribute"
        self.ctx.push()
        try:
            return fn(self, *args, **kwargs)
        finally:
            self.ctx.pop()
    return wrapper
