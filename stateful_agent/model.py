import abc
from langchain.agents import AgentExecutor

class StatefulAgentExecutor(abc.ABC):
    executor: AgentExecutor
    
    @abc.abstractmethod
    def create_agent(self, *args, **kwargs) -> AgentExecutor:
        """Create an agent executor instance"""
        pass

    @abc.abstractmethod
    def terminal_mode(self, executor: AgentExecutor | None = None, *args, **kwargs) -> None:
        """Run the agent_executor in terminal_model"""
        pass