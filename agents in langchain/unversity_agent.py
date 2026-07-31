from langchain_groq import ChatGroq 
from langchain.agents import create_agent 
from langchain_core.tools import tool
from datetime import datetime

GROQ_API_KEY = ""

llm = ChatGroq(api_key=GROQ_API_KEY , model = "llama-3.3-70b-versatile" , temperature=0.3)

@tool 
def calculate_percentage(obtained_marks : float , total_marks: float) -> float:
    """this function returns the percentage of the obtained marks from the total marks"""
    return round((obtained_marks/ total_marks) *100 , 2)


@tool
def gpa_calulator(grades : list[str] ) -> float:
    """Calculate GPA from grades.
    Allowed grades:
    A=4.0
    A-=3.7
    B+=3.3
    B=3.0
    C+=2.3
    C=2.0
    D=1.0
    F=0.0
    """
    mapping = {"A" :4.0 ,"A-":3.7, "B+":3.3, "B":3.0,"C+": 2.3,"C" :2.0, "D" :1.0,"F":0.0 }
    total = sum(mapping[g] for g in grades)
    return round(total/ len(grades) , 2)

@tool 
def timefind()->str:
    """this function returns the current time"""
    return str(datetime.now())

@tool
def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())

@tool
def remaining_days(date :str) -> int:
    """
    Calculate remaining days until a future date.
    Format:
    YYYY-MM-DD
    """
    future = datetime.strptime(date ,"%Y-%m-%d")
    today = datetime.now()
    
    return (future - today).days


tool = [calculate_percentage , gpa_calulator , timefind , word_count , remaining_days]

agent = create_agent(
    model= llm , 
    tools= tool , 
     system_prompt="""
You are an intelligent university assistant.
Help students with:
- GPA
- Percentage
- Assignments
- Time
- Exam dates
Always use the appropriate tool whenever possible.otherwise for other queries u can answer with your knowledge.
Strict rule -> you cannot answer the queries that is out of studies if query is out of studies or university u have to answer 
I can only guide about the University and studies
"""
)

res = agent.invoke(
    {
        "messages": [{
            "role" : "user" , 
            "content" : "Hi my grades in subjects are A , B , C+ tell me my gpa"
        }
        ]
    }
)

print(res['messages'][-1].content)