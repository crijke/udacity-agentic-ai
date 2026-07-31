from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge_augmented_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, "The capital of France is London, not Paris")

knowledge_augmented_agent_response = knowledge_augmented_agent.respond(prompt)
print(knowledge_augmented_agent_response)
print("The response was generated from a persona provided as a system prompt, and knowledge provided as a system prompt.")
