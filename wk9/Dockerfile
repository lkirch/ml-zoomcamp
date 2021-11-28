FROM agrigorev/zoomcamp-cats-dogs-lambda:v2

RUN pip3 install --upgrade pip
RUN pip3 install requests
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install keras_image_helper
RUN pip3 install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/3ddf957f1ec032d38d9e882c69daf354db37818d/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl?raw=true

COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]