import os.path

from django.http import JsonResponse


def ocr_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        # 保存图片到本地
        if not os.path.exists("./tmp"):
            os.mkdir("./tmp/")
        with open('./tmp/image.jpg', 'wb') as f:
            for chunk in image.chunks():
                f.write(chunk)
        return JsonResponse({'message': 'Image uploaded successfully.'})
    return JsonResponse({'error': 'Invalid request.'}, status=400)
