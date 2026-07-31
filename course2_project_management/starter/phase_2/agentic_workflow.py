# agentic_workflow.py

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
with open("Product-Spec-Email-Router.txt", "r") as f:
    product_spec = f.read()

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components\n"
    "When the user asks for the development plan, the features or the development tasks of a product, "
    "always return exactly these three steps, in this order, one step per line, with no introduction, "
    "no explanation and no additional text:\n"
    "1. Define the user stories for the product.\n"
    "2. Group the user stories into product features.\n"
    "3. Define the engineering tasks required to implement the product features."
)
action_planning_agent = ActionPlanningAgent(openai_api_key=openai_api_key, knowledge=knowledge_action_planning)

# Every support function sends the response of its knowledge agent into the matching
# evaluation agent, and that evaluation agent asks the same knowledge agent to respond
# to it once more. The knowledge agents therefore have to recognise an already finished
# answer in their prompt and hand it back, instead of commenting on it in a chat reply.
# This rule is appended as the last part of every knowledge string, so that it is the
# final instruction the agent reads before it answers.
knowledge_answer_handling = (
    "\nMOST IMPORTANT RULE: when the prompt already contains a finished answer, output that answer again in "
    "full. Repeat every single item of it word by word, and only correct or add what is explicitly wrong or "
    "missing. Never answer with a sentence about the answer. Never write that the answer is complete, correct "
    "or covers everything. Never praise, summarise or acknowledge it. Output nothing but the items themselves."
)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    "Product Spec: " + product_spec
    + knowledge_answer_handling
)
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona_product_manager, knowledge=knowledge_product_manager)

# Product Manager - Evaluation Agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria = (
    "The answer must be a list of user stories for the Email Router product, and every single story must "
    "follow this exact structure: 'As a [type of user], I want [an action or feature] so that [benefit/value].'\n"
    "In addition, the content of every story must be substantive and specific to the Email Router product:\n"
    "- The persona must be a real user class of the Email Router (for example Customer Support Representative, "
    "Subject Matter Expert, IT Administrator, Customer).\n"
    "- The action must describe a concrete Email Router capability such as email ingestion, message "
    "classification, knowledge base retrieval, automated response generation, routing to subject matter experts, "
    "the monitoring dashboard, configuration or manual override.\n"
    "- The benefit must state a real value such as faster response times, less manual triage or more consistent answers.\n"
    "Answer 'No' if any story is generic, is unrelated to email handling (for example account creation, login or "
    "password reset), only repeats the field labels, or is a placeholder asking the user for more information."
)
product_manager_evaluation_agent = EvaluationAgent(openai_api_key=openai_api_key, persona=persona_product_manager_eval, evaluation_criteria=evaluation_criteria, worker_agent=product_manager_knowledge_agent, max_interactions=10)


# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for converting user stories into product features."
knowledge_program_manager = (
    "Product features are defined by grouping related user stories into cohesive capabilities of the product. "
    "Every feature must be written with exactly these four labelled fields, in this order:\n"
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user\n"
    "Every feature must be derived from the Email Router user stories provided in the prompt and must describe "
    "real Email Router capabilities such as email ingestion, message classification, knowledge base retrieval, "
    "response generation, routing to subject matter experts, and the monitoring/configuration dashboard. "
    "Never ask the user for the stories and never invent functionality that is not covered by the given stories.\n"
    "Every user story given in the prompt must be covered by exactly one feature, so always produce one feature per "
    "group of related stories and never leave a story uncovered. When you are asked to revise your answer, keep every "
    "feature you already produced and correct only what was criticised. Never reduce the number of features.\n"
    "Product Spec: " + product_spec
    + knowledge_answer_handling
)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona_program_manager, knowledge=knowledge_program_manager)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
program_manager_evaluation_criteria = (
    "The answer must be a list of Email Router product features, and each feature must contain all four of these "
    "labelled fields, in this order:\n"
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user\n"
    "It is not enough that the field labels are present. Each field must contain substantive, Email Router specific "
    "content: the feature must describe an actual product capability such as email ingestion, message classification, "
    "knowledge base retrieval, automated response generation, routing to subject matter experts, or the monitoring "
    "and configuration dashboard, and it must be traceable to the user stories given in the prompt.\n"
    "Features are correctly derived from user stories, so content that originates from user stories is expected and "
    "must NOT be a reason to answer 'No'. Only treat the answer as user stories if it is literally written as "
    "'As a ..., I want ..., so that ...' sentences instead of the four labelled fields.\n"
    "Judge only what is written in the answer. Do not require any specific number of features and do not require "
    "any specific capability to be present: an answer that describes fewer capabilities than the product has is "
    "still acceptable as long as every feature it contains is correctly formed.\n"
    "Answer 'No' only if no feature is present, if any of the four fields is missing, empty, generic or a "
    "placeholder, if the answer asks the user to supply the user stories, if it is a sentence commenting on an "
    "answer instead of the features themselves, or if it describes unrelated functionality such as account "
    "creation, login or password reset."
)
program_manager_evaluation_agent = EvaluationAgent(openai_api_key=openai_api_key, persona=persona_program_manager_eval, evaluation_criteria=program_manager_evaluation_criteria, worker_agent=program_manager_knowledge_agent, max_interactions=10)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for converting product features into detailed engineering tasks."
knowledge_dev_engineer = (
    "Development tasks are defined by identifying what has to be built to implement each product feature and its "
    "underlying user stories. Every task must be written with exactly these seven labelled fields, in this order:\n"
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first\n"
    "Every task must be derived from the Email Router features and user stories provided in the prompt and must "
    "describe implementable engineering work on Email Router capabilities such as SMTP/IMAP/REST email ingestion, "
    "LLM based message classification with confidence scores, the vector knowledge base, the RAG response generation "
    "engine and its approval workflow, the rules based SME routing engine, and the metrics dashboard and configuration "
    "panel. Never ask the user for the stories or features and never invent unrelated work such as account creation, "
    "login or password reset.\n"
    "Define at least one task for every feature and every user story given in the prompt, and cover all capability "
    "areas of the product, not only the first one. When you are asked to revise your answer, keep every task you "
    "already produced and correct only what was criticised. Never reduce the number of tasks.\n"
    "Product Spec: " + product_spec
    + knowledge_answer_handling
)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona_dev_engineer, knowledge=knowledge_dev_engineer)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
persona_dev_engineer_eval_criteria = (
    "The answer must be a list of engineering tasks for the Email Router, and each task must contain all seven of "
    "these labelled fields, in this order:\n"
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first\n"
    "It is not enough that the field labels are present. Each field must contain substantive, Email Router specific "
    "content: the Description must describe implementable work on capabilities such as email ingestion, message "
    "classification, knowledge base retrieval, automated response generation or SME routing; the Related User Story "
    "must reference an actual Email Router user story from the prompt; the Acceptance Criteria must be verifiable; "
    "the Estimated Effort must be a concrete estimate; and Dependencies must name other tasks or state 'None'.\n"
    "Judge only what is written in the answer. Do not require any specific number of tasks and do not require any "
    "specific capability area to be covered: an answer that covers only some areas of the product is still "
    "acceptable as long as every task it contains is correctly formed.\n"
    "Answer 'No' only if no task is present, if any field is missing, empty, generic or a placeholder, if the "
    "answer asks the user to supply the user stories or features, if it is a sentence commenting on an answer "
    "instead of the tasks themselves, or if it describes unrelated work such as account creation, login or "
    "password reset."
)
development_engineer_evaluation_agent = EvaluationAgent(openai_api_key=openai_api_key, persona=persona_dev_engineer_eval, evaluation_criteria=persona_dev_engineer_eval_criteria, worker_agent=development_engineer_knowledge_agent, max_interactions=10)


# Shared state of the workflow. Each support function writes the validated output of
# its stage here, so that the next stage can use it as context for its own prompt.
workflow_state = {
    "user_stories": "",
    "product_features": "",
    "engineering_tasks": "",
}


# ---------------------------------------------------------------------------
# Support function contract
# ---------------------------------------------------------------------------
# Every support function below is the callable that the routing agent invokes for
# one role, and all three follow the same five stage contract:
#
#   1. Input step   - the function receives one step of the action plan as `query`.
#   2. Role prompt  - it builds the prompt for its role by combining that step with
#                     the required output format and, from the second stage on, the
#                     validated output of the previous stages held in workflow_state.
#   3. Worker call  - it explicitly calls respond() on the matching
#                     KnowledgeAugmentedPromptAgent to produce the worker response.
#   4. Validation   - it passes that worker response to evaluate() on the matching
#                     EvaluationAgent, which judges it against the role criteria and
#                     has the worker agent refine it until it is accepted or the
#                     interaction limit is reached.
#   5. Final text   - it stores evaluation_result['final_response'] in workflow_state
#                     and returns that validated text, and nothing else, to the router.
#
# The functions never return the raw worker response and never delegate the worker
# call to the evaluation agent: the chain respond() -> evaluate() -> final_response
# is performed here, in the support function itself.
# ---------------------------------------------------------------------------


def product_manager_support_function(query):
    """Produce the Email Router user stories and store them for the next stage.

    Chain: role prompt -> product_manager_knowledge_agent.respond() ->
    product_manager_evaluation_agent.evaluate() -> validated user stories.
    """
    prompt = (
        f"{query}\n\n"
        "Write the complete set of user stories for the Email Router product described in your knowledge. "
        "Use the format 'As a [type of user], I want [an action or feature] so that [benefit/value].' and cover "
        "the different user classes of the product. Return only the user stories."
    )
    # Step 1: the knowledge agent produces the response for this role.
    response = product_manager_knowledge_agent.respond(prompt)
    # Step 2: the matching evaluation agent validates and, if needed, refines it.
    evaluation_result = product_manager_evaluation_agent.evaluate(response)
    if not evaluation_result["accepted"]:
        print("[Workflow] Warning: the user stories never passed the evaluation criteria.")
    # Step 3: store and return only the validated response.
    workflow_state["user_stories"] = evaluation_result["final_response"]
    return evaluation_result["final_response"]


def program_manager_support_function(query):
    """Convert the user stories from the previous stage into structured product features.

    Chain: role prompt (step + workflow_state['user_stories']) ->
    program_manager_knowledge_agent.respond() ->
    program_manager_evaluation_agent.evaluate() -> validated product features.
    """
    prompt = (
        f"{query}\n\n"
        "Group the Email Router user stories below into product features. "
        "For every feature provide exactly the fields Feature Name, Description, Key Functionality and User "
        "Benefit, each filled with concrete Email Router content. Use only the user stories below as input "
        "and do not ask for additional information.\n\n"
        f"User stories:\n{workflow_state['user_stories']}"
    )
    response = program_manager_knowledge_agent.respond(prompt)
    evaluation_result = program_manager_evaluation_agent.evaluate(response)
    if not evaluation_result["accepted"]:
        print("[Workflow] Warning: the product features never passed the evaluation criteria.")
    workflow_state["product_features"] = evaluation_result["final_response"]
    return evaluation_result["final_response"]


def development_engineer_support_function(query):
    """Convert the product features from the previous stage into detailed engineering tasks.

    Chain: role prompt (step + workflow_state['user_stories'] +
    workflow_state['product_features']) -> development_engineer_knowledge_agent.respond()
    -> development_engineer_evaluation_agent.evaluate() -> validated engineering tasks.
    """
    prompt = (
        f"{query}\n\n"
        "Define the engineering tasks required to implement the Email Router features below. "
        "For every task provide exactly the fields Task ID, Task Title, Related User Story, "
        "Description, Acceptance Criteria, Estimated Effort and Dependencies, each filled with concrete Email "
        "Router content. Reference the actual user stories below in the Related User Story field. Use only the "
        "user stories and features below as input and do not ask for additional information.\n\n"
        f"User stories:\n{workflow_state['user_stories']}\n\n"
        f"Product features:\n{workflow_state['product_features']}"
    )
    response = development_engineer_knowledge_agent.respond(prompt)
    evaluation_result = development_engineer_evaluation_agent.evaluate(response)
    if not evaluation_result["accepted"]:
        print("[Workflow] Warning: the engineering tasks never passed the evaluation criteria.")
    workflow_state["engineering_tasks"] = evaluation_result["final_response"]
    return evaluation_result["final_response"]


# Routing Agent
routing_agent = RoutingAgent(openai_api_key=openai_api_key, agents =[
  {
        "name": "Product Manager",
        "description": (
            "Defines the product requirements and writes the Email Router user stories in the format "
            "'As a [type of user], I want [an action or feature] so that [benefit/value]'. Use this agent for any "
            "step that asks to define, write or extract user stories from the product spec."
        ),
        "func": product_manager_support_function
    },
    {
        "name": "Program Manager",
        "description": (
            "Converts existing user stories into product features. Groups related user stories into cohesive "
            "features and describes each feature with Feature Name, Description, Key Functionality and User Benefit. "
            "Use this agent for any step that asks to define, group or describe product features."
        ),
        "func": program_manager_support_function
    },
    {
        "name": "Development Engineer",
        "description": (
            "Converts product features into detailed engineering tasks. Writes each development task with Task ID, "
            "Task Title, Related User Story, Description, Acceptance Criteria, Estimated Effort and Dependencies. "
            "Use this agent for any step that asks to define engineering or development tasks and the technical "
            "work required to build the product."
        ),
        "func": development_engineer_support_function
    }
])

# Run the workflow

print("\n*** Workflow execution started ***\n")

workflow_prompt = "What would the development tasks for this product be?"

print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")
print("\nDefining workflow steps from the workflow prompt")

action_planning_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
completed_steps = []

for step in action_planning_steps:
    step = step.strip()
    if not step:
        continue
    print(f"\nExecuting step: {step}")

    # Only the short step text is routed, so the routing decision cannot be skewed
    # by long context. The output of the previous stages is carried in workflow_state
    # and added to the prompt inside the support function of the selected agent.
    result = routing_agent.route_prompt(step)
    completed_steps.append(result)
    print(f"Result of step '{step}': {result}")

print("\n*** Workflow execution completed ***\n")

if completed_steps:
    print("\n*** Final Workflow Output ***\n")
    print(completed_steps[-1])

final_plan = "\n\n".join([
    "=" * 80,
    "FINAL EMAIL ROUTER DEVELOPMENT PLAN",
    "=" * 80,
    "--- 1. USER STORIES ---\n\n" + (workflow_state["user_stories"] or "No user stories were produced."),
    "--- 2. PRODUCT FEATURES ---\n\n" + (workflow_state["product_features"] or "No product features were produced."),
    "--- 3. ENGINEERING TASKS ---\n\n" + (workflow_state["engineering_tasks"] or "No engineering tasks were produced."),
    "=" * 80,
    f"END OF PLAN ({len(completed_steps)} workflow steps completed)",
    "=" * 80,
])

print(final_plan)
