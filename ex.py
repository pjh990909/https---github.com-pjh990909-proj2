#STEP 1 추론에 쓸 패키지
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


#STEP 2 추론 만들기
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

#STEP 3 추론 데이터 가져오기
img1 = cv2.imread("iu1.jpg")
img2 = cv2.imread("iu3.jpg")

#STEP 4 추론
faces1 = app.get(img1)
faces2 = app.get(img2)
assert len(faces1)==1
assert len(faces2)==1

print(faces1[0])

#STEP 5
rimg = app.draw_on(img1, faces1)
cv2.imwrite("./iu_output.jpg", rimg)

# then print all-to-all face similarity
# feats = []
# for face in faces:
#     feats.append(face.normed_embedding)
feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
sims = np.dot(feat1, feat2.T)#array를 행렬곱 --> 코사인 유사도  .T를 하면 뒤집어짐(행렬곱을 위해서 뒤집어야함)
print(sims) #0.53089374 유사도를 나타나는거지 같은 사람인지 아닌지 판별이 아님 이 유사도로 타인인지 동일인인지 알아서 설정가능

