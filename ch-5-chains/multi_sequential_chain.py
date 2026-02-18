# insurance_claim_pipeline.py

from typing import Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# Step 0: Initialize the LLM
# -----------------------------
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# -----------------------------
# Step 1: Structured Claim Extraction
# -----------------------------
class ClaimData(BaseModel):
    claimant_name: str = Field(..., description="Name of the person filing the claim")
    claim_type: str = Field(..., description="Type of claim, e.g., 'auto', 'home', 'health'")
    incident_date: str = Field(..., description="Date of the incident YYYY-MM-DD")
    estimated_damage: float = Field(..., description="Estimated damage amount in USD")
    description: str = Field(..., description="Description of the incident")

claim_prompt = ChatPromptTemplate.from_template("""
Extract structured claim data from the following text:

{claim_text}
""")

extract_chain = claim_prompt | model.with_structured_output(ClaimData)

# -----------------------------
# Step 2: Fraud Risk Analysis
# -----------------------------
fraud_prompt = ChatPromptTemplate.from_template("""
Analyze the claim and return JSON with:
- risk_score (0 to 1)
- risk_level (low, medium, high)
- reason (why it is risky or safe)

Claim Data:
{claim_data}
""")

fraud_chain = fraud_prompt | model | JsonOutputParser()

# -----------------------------
# Step 3: Cost Estimation
# -----------------------------
cost_prompt = ChatPromptTemplate.from_template("""
Based on the claim type, estimated damage, and fraud analysis,
estimate the payout amount for the claim. Return JSON with:
- payout_amount (float)
- notes (text explanation)

Claim Data:
{claim_data}

Fraud Result:
{fraud_result}
""")

cost_chain = cost_prompt | model | JsonOutputParser()

# -----------------------------
# Step 4: Internal Adjuster Report
# -----------------------------
report_prompt = ChatPromptTemplate.from_template("""
Generate an internal adjuster report.

Claim Data:
{claim_data}

Fraud Analysis:
{fraud_result}

Cost Estimation:
{cost_result}
""")

report_chain = report_prompt | model | StrOutputParser()

# -----------------------------
# Step 5: Customer Email Generation
# -----------------------------
email_prompt = ChatPromptTemplate.from_template("""
Write a professional email to the claimant explaining their claim decision.
Include payout amount and any important notes. Keep it friendly and clear.

Cost Estimation:
{cost_result}

Fraud Analysis:
{fraud_result}
""")

email_chain = email_prompt | model | StrOutputParser()

# -----------------------------
# Sequential Pipeline Execution
# -----------------------------
def process_claim(claim_text: str):
    # Step 1: Extract structured claim
    claim_data = extract_chain.invoke({"claim_text": claim_text})
    
    # Step 2: Fraud analysis
    fraud_result = fraud_chain.invoke({"claim_data": claim_data.dict()})
    
    # Step 3: Cost estimation
    cost_result = cost_chain.invoke({
        "claim_data": claim_data.model_dump(),
        "fraud_result": fraud_result
    })
    
    # Step 4: Internal report
    internal_report = report_chain.invoke({
        "claim_data": claim_data.model_dump(),
        "fraud_result": fraud_result,
        "cost_result": cost_result
    })
    
    # Step 5: Customer email
    customer_email = email_chain.invoke({
        "cost_result": cost_result,
        "fraud_result": fraud_result
    })
    
    return {
        "claim_data": claim_data.model_dump(),
        "fraud_result": fraud_result,
        "cost_result": cost_result,
        "internal_report": internal_report,
        "customer_email": customer_email
    }

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    claim_text = """
    Nirzar diwan filed a claim for his auto accident on 2026-01-10.
    The car sustained damages estimated at 50 thousand rupees.
    Nirzar claims another driver ran a red light and hit his car from the side.
    """
    
    result = process_claim(claim_text)
    
    print("===== Structured Claim Data =====")
    print(result["claim_data"])
    
    print("\n===== Fraud Analysis =====")
    print(result["fraud_result"])
    
    print("\n===== Cost Estimation =====")
    print(result["cost_result"])
    
    print("\n===== Internal Adjuster Report =====")
    print(result["internal_report"])
    
    print("\n===== Customer Email =====")
    print(result["customer_email"])
