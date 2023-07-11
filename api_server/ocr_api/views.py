import os.path
import shutil

from django.http import JsonResponse

from ocr_api.ocr_infer.predict_system import main


def ocr_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        # 保存图片到本地
        if not os.path.exists("./tmp"):
            os.mkdir("./tmp/")
        with open('./tmp/image.jpg', 'wb') as f:
            for chunk in image.chunks():
                f.write(chunk)
        ocr_text = ocr_infer("./tmp/image.jpg")
        shutil.rmtree("./tmp")
        return JsonResponse({'ocr_text': ocr_text})
    return JsonResponse({'error': 'Invalid request.'}, status=400)


def ocr_infer(image_path):
    return main(image_path)
