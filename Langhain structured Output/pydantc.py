from pydantic import BaseModel , EmailStr , Field
from typing import Optional
class Student(BaseModel):
    name:str = "Ahmad"
    age : Optional[int] = None
    email : EmailStr
    fee : int = Field(gt=5000 , lt=10000 , default= 7000 , description="this is the free of the student for 1st semester ")
new_stu = { "email":"ali@gmail.com", "fee":9999}  # it aumatically know the email
stu = Student(**new_stu)
stu_dict =  dict(stu)
print(stu_dict)
print(stu_dict['email'])