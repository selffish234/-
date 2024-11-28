# 데이터 불러오기
data1 <- read.csv("C:/UnivStudy/UnivLectures/24-2/인공지능기초/과제/텀프로젝트/ML/file/apple_sugar_data2.csv")

# 데이터 확인
str(data)
summary(data)
# 각 값(1, 2, 3)의 개수 확인
table(data$sugar_grade)

# 종속 변수(sugar_grade)를 수치형으로 변환 (필요 시)
data$sugar_grade <- as.numeric(as.factor(data$sugar_grade))  # A, B, C를 각각 1, 2, 3으로 변환

# 다중 회귀 모델 생성
model <- lm(sugar_content_nir ~ soil_ec+ soil_temper +  soil_potential + 
               humidity + sunshine + daylight_hours, data = data)
model3 <- lm(sugar_content_nir ~ soil_ec, data = data)
model4 <- lm(sugar_content_nir ~ soil_ec, data = data)

summary(model3)

# 모델 요약 출력
summary(model)


# 데이터 확인
str(data)
summary(data)

# 종속 변수(sugar_grade)를 수치형으로 변환 (필요 시)
data$sugar_grade <- as.numeric(as.factor(data$sugar_grade))  # A, B, C를 각각 1, 2, 3으로 변환

# 독립변수 선택
independent_vars <- c("soil_temper", "soil_humidity", "soil_potential", 
                       "humidity", "sunshine", "daylight_hours")

# 독립변수 정규화
data2 <- scale(data[independent_vars])
# data2를 데이터 프레임으로 변환
data2 <- as.data.frame(data2)

# 변환 확인
class(data2)

# data2에 data의 sugar_content_nir 열 추가
data2$sugar_content_nir <- data$sugar_content_nir

# 다중 회귀 모델 생성
model2 <- lm(sugar_content_nir ~ soil_temper + soil_humidity + soil_potential + 
             humidity + sunshine + daylight_hours, data = data2)

# 모델 요약 출력
summary(model2)

