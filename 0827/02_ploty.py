# 붓꽃(iris) 데이터를 Plotly 3차원 산점도로 시각화하는 예제

import plotly.express as px  # Plotly Express: 적은 코드로 대화형 그래프를 만드는 고수준 API

# 내장 예제 데이터셋(iris)을 pandas DataFrame으로 불러온다.
# 컬럼 구성: sepal_length(꽃받침 길이), sepal_width(꽃받침 너비),
#            petal_length(꽃잎 길이), petal_width(꽃잎 너비), species(품종)
df = px.data.iris()

# 3차원 산점도 생성
#   x, y, z : 3개의 축에 매핑할 수치형 특성
#   color   : 품종(species)별로 점 색을 다르게 표시 -> 군집 구분이 쉬워진다
# 특성은 4개지만 3차원 공간에는 3개까지만 표현할 수 있으므로 petal_length를 제외한다.
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', color='species') # petal_length를 제외하여 3차원 공간 구성

# 그래프를 기본 웹 브라우저 창에 띄운다.
# renderer="browser"를 지정하면 IDE 내부가 아닌 브라우저에서 열려
# 마우스로 회전/확대하며 3차원 분포를 살펴볼 수 있다.
fig.show(renderer="browser")
