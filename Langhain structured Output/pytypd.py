from typing import TypedDict

# we have to define the class in which we need output in structured form 
class person(TypedDict):
    name :str
    age : int


new_person : person = {"name":"Hunain ", "age":"18" }  # if we define it as an int we can also give it as a string 
print(new_person)