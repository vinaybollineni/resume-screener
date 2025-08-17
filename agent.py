import os
import logging
import sys
from typing import Optional
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# --- NEW IMPORTS ---
try:
    from langchain_ollama import OllamaLLM
    from langchain_ollama import OllamaEmbeddings
    OllamaLLM.__name__
    OllamaEmbeddings.__name__
except ImportError:
    print("Error: The 'langchain-ollama' package is not installed or the import failed.")
    print("Please run `pip install -U langchain-ollama` and try again.")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- TOOL DEFINED OUTSIDE THE CLASS FOR ROBUSTNESS ---
# We make this a standalone function to avoid the complex 'self' handling
# by the agent, which caused the validation error.
@tool
def find_candidates(job_description: str) -> str:
    """Searches through the resumes for candidates that match the job description."""
    try:
        # Load the FAISS vector store on each call, this is fine for small local stores
        DB_FAISS_PATH = "local_store/faiss_index"
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )

        results = vectorstore.similarity_search(job_description, k=5)
        
        if not results:
            return "No matching candidates found for the given job description."
        
        candidate_info = []
        for i, doc in enumerate(results, 1):
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            candidate_info.append(f"Candidate {i}:\nSource: {source}\nContent:\n{content}")
            
        return f"Found {len(results)} matching candidates:\n\n" + "\n\n".join(candidate_info)
        
    except Exception as e:
        logger.error(f"Error in find_candidates: {e}")
        return f"Error occurred while searching for candidates: {str(e)}"
    
# --- START OF CLASS ---
class ResumeScreenerAgent:
    def __init__(self):
        self.llm: Optional[OllamaLLM] = None
        self.embeddings: Optional[OllamaEmbeddings] = None
        self.agent: Optional[AgentExecutor] = None
        
    def load_models(self) -> bool:
        """Load the LLM and embedding models."""
        try:
            try:
                test_llm = OllamaLLM(model="mistral")
                test_llm.invoke("test")
                
                test_embeddings = OllamaEmbeddings(model="nomic-embed-text")
                test_embeddings.embed_query("test")

                logger.info("Ollama models are available and working.")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama models. Check if Ollama is running.")
                logger.error(f"Error: {e}")
                return False
                
            self.llm = OllamaLLM(model="mistral")
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
            logger.info("Models loaded successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def initialize_agent(self) -> bool:
        """Initialize the LangChain agent."""
        try:
            if not self.llm:
                logger.error("Cannot initialize agent: models not loaded.")
                return False
            
            # The tool is now a global function, so we just pass it
            tools = [find_candidates]
            
            # Get the prompt from LangChain Hub
            prompt = hub.pull("hwchase17/react")
            
            self.agent = AgentExecutor(
                agent=create_react_agent(self.llm, tools, prompt),
                tools=tools,
                verbose=True,
                handle_parsing_errors=True
            )
            
            logger.info("Agent initialized successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            return False
    
    def run(self, job_description: str) -> str:
        """Run the agent with the given job description using the new invoke method."""
        try:
            if not self.agent:
                return "Error: Agent not initialized."
            
            logger.info("Agent is processing the job description...")
            
            response = self.agent.invoke(
                {"input": f"Find me 1 candidate who is a great fit for this role:\n{job_description}"}
            )

            result = response['output']

            return result
            
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            return f"Error occurred while running the agent: {str(e)}"
    
    def setup(self) -> bool:
        """Complete setup of the agent system."""
        logger.info("Setting up Resume Screener Agent...")
        
        if not self.load_models():
            return False
            
        # The vector store is now loaded inside the tool, so no need to call it here
        if not self.initialize_agent():
            return False
            
        logger.info("Resume Screener Agent setup completed successfully.")
        return True

def main():
    """Main function to run the resume screener."""
    agent_system = ResumeScreenerAgent()
    
    # We still need to call the ingestion script once to create the vector store
    # This check ensures it exists
    DB_FAISS_PATH = "local_store/faiss_index"
    if not os.path.exists(DB_FAISS_PATH):
        logger.error("FAISS index directory not found. Please run ingest.py first.")
        sys.exit(1)

    if not agent_system.setup():
        logger.error("Failed to setup the agent system. Exiting.")
        sys.exit(1)
    
    try:
        job_desc = input("Enter the job description for a candidate: ").strip()
        
        if not job_desc:
            logger.error("Job description cannot be empty.")
            return
        
        print("\n\nAgent is thinking...\n")
        result = agent_system.run(job_desc)
        print(f"\nResult:\n{result}")
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
