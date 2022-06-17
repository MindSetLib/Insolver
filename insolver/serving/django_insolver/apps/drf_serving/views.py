import json

import pandas as pd
from drf_serving.load_model import model, tranforms
from insolver import InsolverDataFrame
from insolver.transforms import InsolverTransform
from rest_framework.views import APIView, Response


class PredictAPIView(APIView):
    def post(self, request):
        # json request
        json_str = json.dumps(request.data['df'])
        df = pd.read_json(json_str)
        insdataframe = InsolverDataFrame(df)

        # Apply transformations
        instransforms = InsolverTransform(insdataframe, tranforms)
        instransforms.ins_transform()

        # Prediction
        predicted = model.predict(instransforms)

        result = {
            'predicted': predicted.tolist()
        }
        return Response(result)
