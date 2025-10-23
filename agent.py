import asyncio
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from typing import Any
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.agent.workflow import AgentWorkflow
from github import Github, Auth
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context


class GithubTools:
    """
    A utility class designed for interacting with GitHub repositories and pull requests
    using a GitHub client object. This class encompasses methods to retrieve pull request
    details, commit information, file contents, and manipulate context state for integration
    purposes.

    This class simplifies GitHub operations by abstracting common tasks while providing
    capabilities for retrieving and saving GitHub data.

    :ivar _repo: The repository instance retrieved from the GitHub client
                 using the repository name or identifier.
    :type _repo: Repository
    """
    
    def __init__(self, github_client: Github, repo_name: str |
                                                         int):
        self._g = github_client
        self._repo_name = repo_name
        self._repo = github_client.get_repo(repo_name)
    
    async def get_pr_details(self, pr_number: int) -> dict:
        """
           Get PR information: author, title, body, state, commit SHAs.

           IMPORTANT: The 'body' field contains PR description (unreliable for file names).
           To get actual changed files, you MUST call pr_commit_detail with each SHA from head_sha.

           Returns:
               dict with author, title, body, diff_url, state, head_sha (list of commit SHAs)
        """
        
        pr = self._repo.get_pull(pr_number)
        info_dict = {
            "user": pr.user,
            "title": pr.title,
            "body": pr.body,
            "diff_url": pr.diff_url,
            "state": pr.state,
            "commit_SHAs": [c.sha for c in pr.get_commits()],
        }
        return info_dict
    
    def get_pr_commit_detail(self, head_sha: str) -> list[dict[str, Any]]:
        """
        Given the commit SHA, this function can retrieve information about
        the commit, such as the files that changed, and return that information.
        """
        commit = self._repo.get_commit(head_sha)
        changed_files: list[dict[str, Any]] = []
        for f in commit.files:
            changed_files.append({
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,
            })
        return changed_files if changed_files else None
    
    def get_file_content(self, file_path: str, ref: str) -> str | None:
        """
        Get file content from a given ref
        """
        try:
            content = self._repo.get_contents(file_path, ref=ref)
            if isinstance(content, list):
                return None
            
            if content.type != "file":
                return None
            
            return content.decoded_content.decode("utf-8")
        except Exception:
            return None
    
    async def add_gathered_context_to_state(self, ctx: Context,
                                            gathered_context: str) \
            -> None:
        """
        Save gathered context to state, to give a chance to other agents to use it.
        """
        current_state = await ctx.store.get("state")  # type: ignore
        current_state["gathered_context"] = gathered_context
        await ctx.store.set("state", current_state)  # type: ignore
    
    async def add_draft_comment(self, ctx: Context, draft_comment: str) \
            -> None:
        """
        Save context to state to give a chance to other agents to use it.
        """
        current_state = await ctx.store.get("state")  # type: ignore
        current_state["draft_comment"] = draft_comment
        await ctx.store.set("state", current_state)  # type: ignore
    
    async def add_final_review_to_state(self, ctx: Context,
                                        final_review_comment: str) -> None:
        """
        Save your final review to the state.
        """
        current_state = await ctx.store.get("state")  # type: ignore
        current_state["final_review_comment"] = final_review_comment
        await ctx.store.set("state", current_state)  # type: ignore
    
    async def post_final_review_to_github_pr(self, ctx: Context,
                                             pr_number: int) \
            -> None:
        """
        Post your final review to the PR comment on Github.
        """
        current_state = await ctx.store.get("state")  # type: ignore
        self._repo.get_pull(pr_number).create_review(body=current_state.get(
            "final_review_comment"))  # type: ignore
    
    def to_function_tools(self, method_names: list[str] | None = None) -> list[
        FunctionTool]:
        tools = []
        for name in method_names:
            method = getattr(self, name, None)
            if method and callable(method):
                tools.append(FunctionTool.from_defaults(method))
        
        return tools


def create_agents(llm: OpenAI, github_tools):
    """"""
    
    tools_context_agent = github_tools.to_function_tools(["get_pr_details",
                                                          "get_file_content",
                                                          "get_pr_commit_detail",
                                                          "add_gathered_context_to_state"])
    
    context_agent = FunctionAgent(
        name="ContextAgent",
        tools=tools_context_agent,
        llm=llm,
        verbose=True,
        description="Gathers all the needed context and save it to the state.",
        can_handoff_to=["CommentorAgent", "ReviewAndPostingAgent"],
        system_prompt="""
        You are the context gathering agent. When gathering context, you MUST gather \n:
      - The details: author, title, body, diff_url, state, and head_sha; \n
      - Changed files; \n
      - Any requested for files; \n
        Once you gather the requested info, you MUST hand control back to the Commentor Agent.
        """
    )
    
    commentor_agent = FunctionAgent(
        name="CommentorAgent",
        system_prompt='''
        You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n
    Ensure to do the following for a thorough review:
     - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
        - If you need any additional details, you must hand off to the
     ContextAgent, Do NOT ask user!. \n
     - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
        - What is good about the PR? \n
        - Did the author follow ALL contribution rules? What is missing? \n
        - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
        - Are new endpoints documented? - use the diff to determine this. \n
        - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
     - You should directly address the author. So your comments should sound like: \n
     "Thanks for fixing this. I think all places where we call quote should
     be fixed. Can you roll this fix out everywhere?".\n
     - You must hand off to the ReviewAndPostingAgent once you are done drafting a review.
        ''',
        llm=llm,
        verbose=True,
        description="Uses the context gathered by the context agent to draft a pull review comment comment.",
        tools=github_tools.to_function_tools(["add_draft_comment"]),
        can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
    )
    
    review_and_posting_agent = FunctionAgent(
        name="ReviewAndPostingAgent",
        system_prompt="""
        You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.
        """,
        llm=llm,
        verbose=True,
        description="Posts a review to GitHub once it is ready.",
        tools=github_tools.to_function_tools(
            ["add_final_review_to_state", "post_final_review_to_github_pr"]),
        can_handoff_to=["CommentorAgent"]
    
    )
    
    return context_agent, commentor_agent, review_and_posting_agent

repository = os.getenv("REPOSITORY")
repo_url = "https://github.com/MBorky/recipes-api.git"
load_dotenv(override=True)
git_token = os.getenv("GITHUB_TOKEN")
base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")
auth = Auth.Token(git_token)
g = Github(auth=auth)
pr_n = int(os.getenv("PR_NUMBER"))

llm = OpenAI(model="gpt-4o-mini", api_base=base_url, api_key=api_key)
github_tools = GithubTools(g, repository)

context_agent, commentor_agent, review_and_posting_agent = create_agents(github_tools=github_tools,
                                                llm=llm)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_context": "",
        "pr_number": "",
        "draft_comment": "",
        "final_review_comment": "",
    },
)


async def main():
    query = f"Write a review for PR number {pr_n}"
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    repo_url = "https://github.com/MBorky/recipes-api.git"
    asyncio.run(main())
    g.close()