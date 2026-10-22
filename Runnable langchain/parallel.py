from langchain_core.runnables import RunnableSequence , RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from typing import  TypedDict,Annotated 
import sys

sys.stdout.reconfigure(encoding="utf-8")


load_dotenv()

model1 = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite" , temperature = 0.8)
model2 = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")
parser=StrOutputParser()


prompt1 = PromptTemplate(
    template="Generate me the caption for my linkedin account strict rule donot add emojes   form the given text :- \n{text}" , 
    input_variables=['text']
    
)

prompt2 = PromptTemplate(
    template="Generate me the caption for my twitter account strict rule donot add emojis form the given text :- \n{text}" , 
    input_variables=['text']
    
)

chain = RunnableParallel(
    {
        "linkedin" : prompt1 | model1 | parser ,
        "twitter"  : prompt2 | model2 | parser
    }
)
result = chain.invoke({"text":"XGBoost (Extreme Gradient Boosting) is a powerful supervised machine learning algorithm widely used for both classification and regression tasks. It is based on the gradient boosting framework, where multiple decision trees are built sequentially, and each new tree attempts to correct the errors made by the previous ones. The algorithm uses gradient descent to minimize the loss function and incorporates both L1 (Lasso) and L2 (Ridge) regularization techniques to reduce overfitting and improve generalization. XGBoost can automatically handle missing values, supports parallel computation for faster training, and scales efficiently to large datasets. It also provides feature importance scores, helping users understand which features contribute most to predictions. Hyperparameters such as the learning rate, maximum tree depth, and number of estimators can be tuned to optimize performance, and early stopping can be used to prevent overfitting. XGBoost is particularly effective on structured and tabular data and has consistently achieved top results in many Kaggle machine learning competitions. It is widely applied in industries such as finance, healthcare, marketing, and fraud detection due to its high accuracy performance."})

print(result["linkedin"])
print("---------------------------------------------------------------------------------------")
print(result["twitter"])