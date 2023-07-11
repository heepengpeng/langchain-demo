# Create your views here.
import json

from rest_framework.decorators import api_view
from rest_framework.response import Response


@api_view(['POST'])
def save_contract(request):
    if request.method == 'POST':
        # 获取JSON数据
        json_data = request.data

        # 存储到文件
        try:
            with open('contract_extract.json', 'w', encoding="utf-8") as file:
                json.dump(json_data, file, indent=4, ensure_ascii=False)
        except IOError:
            return Response('Contract save failed', status=500)

        return Response('Contract save successfully')
    else:
        return Response('Invalid request method', status=405)
