# House Prices - Advanced Regression Techniques

Competition: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

# Preprocessing

## 1. Handling N/As
პირველ რიგში, შევხედე `data-description.txt` ფაილს და column-ებს თვალი გადავავლე.

1. **ზოგი column-ის N/A ჩავანაცვლე 'None'-ით.**  
ზოგ column-ში არსებული N/A რეალურად N/A არ არის, და კონკრეტული column რა მნიშვნელობასაც ატარებს, მაგის არარსებობას ნიშნავს. მაგალითად, BsmtQual-ში თუ გვხვდება N/A, ნიშნავს რომ Basement არ აქვს მაგ კონკრეტულ სახლს. ამიტომ, საჭიროა რომ ეგ N/A value-ები ჩანაცვლდეს რაღაც შესაბამისით.  
2. **ზოგი column-ის N/A ჩავანაცვლე შესაბამისი 0-ით.**  
ამის შემდეგ, შევხედე ისეთ column-ებს, რომელთა მნიშვნელობაც შესაძლოა დალინკული ყოფილიყო რომელიმე სხვა column-თან, ანუ თუ იმ სხვა column-ის მნიშვნელობა 'None' იქნებოდა, მაშინ ამ დამოკიდებული column-ის მნიშვნელობაც ბუნებრივია, რომ 0 უნდა იყოს. ასეთი column-ების მაგალითია `MasVnrArea <- MasVnrType`, `BsmtFinSF1 <- BsmtFinType1` და სხვა.
3. **დანარჩენ column-ებს ვეპყრობი ტიპის შესაბამისად.**  
თუ categorical column-ია, ვანაცვლებ მოდით, თუ numerical column-ია, ვანაცვლებ მედიანით.

## 2. Dropping irrelevant columns
არის გარკვეული column-ები, რომლებზეც ვფიქრობ რომ prediction-ში დიდ წვლილს არ შეიტანს.

1. **Street, Utilities, PoolQC**  
ამ კოდის საშუალებით:
```py
# examine distribution of each categorical column
for catcol in cat_cols:
  plt.figure(figsize=(6, 4))
  sns.countplot(data=X_train, x=catcol)
  plt.title(f'{catcol}')
  plt.show()
```
შევხედე კატეგორიულ ცვლადებში მნიშვნელობების განაწილებას. ზემოთ ხსენებულ 3 column-ში თითქმის 100%-ით მხოლოდ ერთი value გვხვდებოდა, ამიტომ ჩავთვალე, რომ მათ feature-ებად ქონას აზრი არ ქონდა. (Pool-თან დაკავშირებული feature მაინც შევინარჩუნე რადგან Pool-ის არსებობა ჩავთვალე რომ სახლს ფასს უმატებს, ქვემოთ ვახსენებ მაგ feature-ს.)

2. **MoSold**  
ვფიქრობ რომ რომელ თვეში გაიყიდა სახლი არაა მნიშვნელოვანი, მხოლოდ წელი.

3. **YearBuilt**  
Dataset-ში გვხვდება YearRemodAdd column, რისი მნიშვნელობაც YearBuilt-ს უდრის მაშინ, როცა რემოდელინგი არ ჩატარებია სახლს, თუარადა ბოლოს როდესაც ჩაუტარდა იმ წელს. რახან რემოდელინგი აზრობრივად განახლებას ნიშნავს სახლის და გაუმჯობესებას, ვფიქრობ რომ მთავარი ყურადღება მაინც ამ feature-ს მიექცევა და შემიძლია რომ YearBuilt საერთოდ ამოვიღო.

## 3. Explicit Conversions
`data-descriptions.txt` ფაილში column-ებზე დაკვირვებით შევამჩნიე:

რომ ზოგი column უბრალოდ სიკარგის დონეს განსაზღვრავს და შესაბამისი კატეგორიული მნიშვნელობების რიცხვითში გადაყვანა ლოგიკური იქნება. მაგალითად, `Ex(cellent) = 10`, `Gr(eat) = 8` და ა.შ.

ასევე, column-ები (Condition1 და Condition2), (Exterior1st და Exterior2nd) შეიძლება გადაითარგმნოს column მატრიცაში და ყველა კატეგორიისთვის `has_კატეგორია` feature შეიქმნას.

ასევე, MSSubClass რეალურად კატეგორიული column-ია.

## 4. Feature Engineering
ისევ column-ებზე დაკვირვებით, შევქმენი ახალი column-ები, რომელთა არსებობასაც ვთვლი რომ ლოგიკურია და სახლისთვის ღირებული რაღაცის ქონაზე მიუთითებს, მაგალითად, `has_pool` - როგორც ზემოთ ვახსენე, აუზის არსებობა სახლს მგონია რო ფასს უმატებს. ეს feature გამოვიყვანე PoolArea-დან. თუ PoolArea > 0, ესეიგი სახლს აუზი აქვს, ან პირიქით. დანარჩენი feature-ები მარტივად წასაკითხად მაქვს ჩამოწერილი `FeatureEngineerPreprocessor` კლასში.

## 5. Categoric to numeric conversions
ზემოთ ნახსენები ხელით გადაყვანები იდეაში OrdinalEncoding-ია. ამ ეტაპზე ვამატებ One-Hot Encoding-ს.

## 6. Correlation Filter
შემდგომ, dataset-ს ვატარებ კორელაციის ფილტრში მხოლოდ ოროგინალ და Feature Engineering-ის დროს მიღებულ column-ებზე. ვსორტავ და თითო-თითოდ ვ-drop-ავ column-ებს წყვილიდან იმისდა მიხედვით თუ რომელია შედარებით ნაკლებად კორელირებული target variable-სთან.

## 7. RFE
შესაბამისი მოდელისთვის ვიყენებ RFE-ს რომ ყველაზე მნიშვნელოვანი feature-ები დავიტოვო.

# Training
პირველ რიგში, დავა-define-ე pipeline, რომელშიც გავაერთიანე ყველა ზემოთ ხსენებული preprocessor-ი, რომელიც ყველა მოდელისთვის საერთოა. ცალკე დავლოგე ეს pipeline mlflow-ში და მოდელადაც დავამატე რეგისტრში რომ მერე `test.csv`-ზე submission-ის დასაგენერირებლად ჩამომეტვირთა და გამეშვა. ყველა მოდელს რომელიც ვცადე აქვს თავისი შესაბამისი pipeline და grid search-ს გადის ამ pipeline-ებში მოქცეული ტრანსფორმერების ჰიპერპარამეტრებზე. ჰიპერპარამეტრების სავარაუდო მნიშვნელობები პირველივე გარტყმით არ მიწერია, სხვადასხვა მნიშვნელობებს ვცდიდი და საბოლოოდ დემონსტრაციისთვის ისეთი პარამეტრებიც მაქვს დატოვებული, რომლებმაც ყველაზე მაღალი შედეგი მომცა.

მოდელები:
1. LinearRegression. train-სა და test-ზე მნიშვნელოვნად აცდენილი score-ები დამიბრუნა. Kaggle როგორც ითვლის ისე მაქვს კოდშიც scoring function შექმნილი. ამ აცდენის მიზეზად overfitting-ს ვასახელებ, და solution-ად რეგულარიზაცია მიმაჩნია.
2. RidgeRegression, LassoRegression. L2 და L1 რეგულარიზირებული მოდელები, რომლებსაც alpha დავუტუნინგე. დაახლოებით ერთნაირი პასუხები მომცა, თუმცა Linear-თან შედარებით გაუმჯობესებული იყო ორივე ცალსახად.
3. RandomForestRegression. დავალების წერის დროს ლექციაზე ახსნილი არ მქონდა კარგად ნასწავლი, ვიდეოებიც არ მინახავს. ამიტომ, ჰიპერპარამეტრები და-google-ული მაქვს და ისე მაქ რამდენიმე არჩეული.

საბოლოოდ, შედარების საფუძველზე ვამბობ რომ საუკეთესო შედეგი Lasso-ს ქონდა, რაზეც Kaggle-ს submission-ის score-იც მეტყველებს (დანარჩენებთან შედარებით).

ჰიპერპარამეტრების ოპტიმიზაციისთვის, როგორც ზემოთ ვახსენე, GridSearchCV მაქვს გამოყენებული.

# Repository Structure
`model_experiment.ipynb` - primary სამუშაო ადგილი, თუმცა რაღაცეები რასაც მუშაობისას ვაკეთებდი ამოღებული მაქვს რომ ძაან არ გადამეტვირთა და ადვილი წასაკითხი ყოფილიყო notebook.  
`model_inference.ipynb` - აქ ვა-load-ებ model registry-დან პრეპროცესორსა და საუკეთესო მოდელს, ვაგენერირებ submission-ს.

# MLFlow
ექსპერიმენტებად:

1. [Preprocessor](https://dagshub.com/b3tameche/kaggle-house-prices.mlflow/#/experiments/0)
2. [Linear Regression](https://dagshub.com/b3tameche/kaggle-house-prices.mlflow/#/experiments/6)
3. [Ridge Regression](https://dagshub.com/b3tameche/kaggle-house-prices.mlflow/#/experiments/7)
4. [Lasso Regression](https://dagshub.com/b3tameche/kaggle-house-prices.mlflow/#/experiments/8)
5. [Random Forest](https://dagshub.com/b3tameche/kaggle-house-prices.mlflow/#/experiments/9)

თითო მოდელის ექსპერიმენტში დალოგილი მაქ grid search-ის ჰიპერპარამეტრების pool-ი, შერჩეული საუკეთესო პარამეტრები და RMSE score-ები train-ზეც და test-ზეც, და თვითონ საუკეთესო pipeline-ები.