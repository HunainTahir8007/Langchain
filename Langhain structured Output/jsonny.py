from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict , Annotated ,Optional  , Literal
from pydantc import BaseModel , EmailStr , Field 
load_dotenv()


model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite") 
summer = {
    "type": "object",
    "description": "Fetch the required thing from query",
    "properties": {   
        "summary": {
            "type": "string",
            "description": "Clear summary of the paragraph"
        },
        "rating": {
            "type": "integer",  
            "description": "Give the rating (1-5)"
        },
        "sentiment": {
            "type": "string",
            "enum": ["Positive", "Negative"],
            "description": "Give the sentiment"
        }
    },
    "required": ["summary", "rating", "sentiment"]
}
 
structured_model = model.with_structured_output(summer)

response = structured_model.invoke("""The iPhone 16 is widely regarded as one of the best standard smartphones released by Apple. It delivers excellent performance thanks to the A18 chip, making everyday tasks, gaming, and multitasking smooth and responsive. The camera system has also been improved, producing high-quality photos and videos, especially in daylight and low-light conditions. Battery life is another strong point, with most users reporting that it easily lasts an entire day on a single charge.

The phone features a premium design with a bright OLED display, improved durability, and useful additions such as the Action Button and Camera Control. Apple also provides long-term software updates, making the device a good investment for users who plan to keep their phone for several years.

However, the iPhone 16 has some drawbacks. The most common criticism is that it still uses a 60 Hz display, while many competing Android smartphones in the same price range offer smoother 120 Hz displays. Additionally, it lacks a dedicated telephoto camera, so its zoom capabilities are not as advanced as those of the Pro models. Users upgrading from the iPhone 15 may also find the improvements relatively minor.

Overall, the iPhone 16 receives highly positive reviews for its performance, camera quality, battery life, and software experience. It is considered an excellent choice for users upgrading from older iPhones or those seeking a premium smartphone without the higher cost of the Pro models.for any query contact us at appleofficial.email.com""")
response = dict(response)
print(response["summary"])
print(response['rating'])
print(response['sentiment'])
