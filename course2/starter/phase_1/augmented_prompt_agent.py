import os
from dotenv import load_dotenv
from workflow_agents.base_agents import AugmentedPromptAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

augmented_agent_response = augmented_agent.respond(prompt)

print(augmented_agent_response)
print("The response was generated from knowledge of the model and a persona provided as a system prompt.")
