# In case of errors install: pip install psutil kaleido

from base64 import b64decode, b64encode
from io import BytesIO

import plotly
import plotly.graph_objects as go
from PIL import Image

fig = go.Figure(data=go.Bar(y=[2, 3, 1]))

fig = plotly.io.to_image(fig, width=1000, format='jpeg')
fig_base64 = b64encode(fig).decode('ascii')
print('Base64 encoded image:\n', fig_base64)

img = Image.open(BytesIO(b64decode(fig_base64)))
img.show()
