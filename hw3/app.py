"""SecOps Sidekick: a multi-agent security research assistant.

A supervisor agent delegates work to specialist sub-agents, each of which
owns its own set of tools:

    research_agent  -- DuckDuckGo web search and Wikipedia (built-in tools)
    security_agent  -- NVD CVE lookup/search and DNS lookup (custom tools)
    coding_agent    -- Python REPL for calculation and data wrangling
    database_agent  -- read-only SQL toolkit over a SQLite file (only when
                       a database is given with --db)

Usage:
    uv run app.py [--db FILE]                 start an interactive session
    uv run app.py [--db FILE] "<question>"    answer one question and exit
    uv run app.py -h                          show this message

    --db FILE   SQLite database to expose to the database agent, for
                example a ROADrecon export (roadrecon.db).

Environment variables:
    GOOGLE_API_KEY     API key for Gemini (required).
    GOOGLE_MODEL       Gemini model name, as in the labs (required).
    NVD_API_KEY        Optional NVD key; only raises the NVD rate limit.
    HW3_AUTO_APPROVE   Set to 1 to run agent-written Python without asking.

Classes:
    CveLookupInput     Validated input schema for cve_lookup.

Functions:
    cve_lookup         Fetch one CVE record from the NVD.
    cve_search         Search the NVD for CVEs by keyword.
    dns_lookup         Resolve DNS records for a hostname.
    run_python         Run Python code in a PythonREPL after user approval.
    build_supervisor   Assemble the supervisor and its sub-agents.
    ask                Send one message to the supervisor and return its reply.
    main               Command line entry point.
"""

import argparse
import os
import re
import sys

import dns.exception
import dns.resolver
import requests
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import SQLDatabase, WikipediaAPIWrapper
from langchain_experimental.utilities import PythonREPL
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field, field_validator

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
DNS_RECORD_TYPES = ("A", "AAAA", "CNAME", "MX", "NS", "TXT")

_python_repl = PythonREPL()


# --------------------------------------------------------------------------
# Custom tools
# --------------------------------------------------------------------------


def _nvd_get(params: dict) -> dict:
    """Query the NVD CVE API and return the decoded JSON response.

    Arguments:
    params -- query string parameters for the NVD CVE 2.0 endpoint

    Raises requests.RequestException on network or HTTP errors.
    """
    headers = {}
    if os.environ.get("NVD_API_KEY"):
        headers["apiKey"] = os.environ["NVD_API_KEY"]
    response = requests.get(NVD_URL, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def _summarize_cve(cve: dict) -> str:
    """Return a short text summary of one NVD CVE record.

    Arguments:
    cve -- the "cve" object of an entry in an NVD response
    """
    description = next(
        (d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"),
        "No description available.",
    )
    severity = "not scored"
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if metrics.get(key):
            data = metrics[key][0]["cvssData"]
            label = data.get("baseSeverity") or metrics[key][0].get("baseSeverity", "")
            severity = f"CVSS {data.get('version')} {data.get('baseScore')} {label}".strip()
            break
    published = cve.get("published", "unknown")[:10]
    return f"{cve.get('id')} (published {published}, {severity}): {description}"


class CveLookupInput(BaseModel):
    """Input schema for cve_lookup; rejects malformed CVE identifiers."""

    cve_id: str = Field(description="CVE identifier such as CVE-2021-44228")

    @field_validator("cve_id")
    @classmethod
    def is_cve_id(cls, value: str) -> str:
        """Return the identifier upper-cased, or raise ValueError if malformed."""
        value = value.strip().upper()
        if not re.fullmatch(r"CVE-\d{4}-\d{4,}", value):
            raise ValueError("Malformed CVE identifier; expected CVE-YYYY-NNNN")
        return value


@tool(args_schema=CveLookupInput)
def cve_lookup(cve_id: str) -> str:
    """Look up one vulnerability in the NIST National Vulnerability Database.

    Use this when the user names a specific CVE identifier.
    """
    try:
        data = _nvd_get({"cveId": cve_id})
    except requests.RequestException as error:
        return f"NVD request failed: {error}"
    items = data.get("vulnerabilities", [])
    if not items:
        return f"No NVD record found for {cve_id}."
    return _summarize_cve(items[0]["cve"])


@tool
def cve_search(keyword: str, limit: int = 5) -> str:
    """Search the NIST National Vulnerability Database by keyword.

    Use this to find CVEs affecting a product or technology when no CVE
    identifier is known.

    Args:
        keyword: Product or technology to search for, such as "Exchange Server"
        limit: Maximum number of CVEs to return (1 to 20)
    """
    limit = max(1, min(limit, 20))
    try:
        data = _nvd_get({"keywordSearch": keyword, "resultsPerPage": limit})
    except requests.RequestException as error:
        return f"NVD request failed: {error}"
    items = data.get("vulnerabilities", [])
    if not items:
        return f"No CVEs found for '{keyword}'."
    total = data.get("totalResults", len(items))
    lines = [_summarize_cve(item["cve"]) for item in items]
    return f"{total} total matches; showing {len(lines)}:\n" + "\n".join(lines)


@tool
def dns_lookup(hostname: str, record_type: str = "A") -> str:
    """Resolve DNS records for a hostname.

    Args:
        hostname: Domain name to resolve, such as example.com
        record_type: One of A, AAAA, CNAME, MX, NS, or TXT
    """
    record_type = record_type.strip().upper()
    if record_type not in DNS_RECORD_TYPES:
        return f"Unsupported record type; choose one of {', '.join(DNS_RECORD_TYPES)}."
    try:
        answers = dns.resolver.resolve(hostname.strip(), record_type, lifetime=10)
    except dns.exception.DNSException as error:
        return f"DNS lookup failed: {error}"
    return "\n".join(answer.to_text() for answer in answers)


@tool
def run_python(code: str) -> str:
    """Run Python code in a persistent REPL and return what it printed.

    Use this for arithmetic, statistics, parsing, and data manipulation.
    Only printed output is returned, so print() the values you need.

    Args:
        code: Python source code to execute
    """
    if os.environ.get("HW3_AUTO_APPROVE") != "1":
        print(f"\n--- The agent wants to run this Python code ---\n{code}\n---")
        if input("Run it? [y/N] ").strip().lower() != "y":
            return "The user declined to run this code."
    output = _python_repl.run(code)
    return output if output.strip() else "(the code ran but printed nothing)"


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------


def _show_step(step: dict) -> None:
    """Print the tool calls and tool results contained in one stream step.

    Arguments:
    step -- one update yielded by agent.stream()
    """
    for message in (step.get("model") or {}).get("messages", []):
        for call in message.tool_calls:
            print(f"🛠️  Tool Call: {call['name']}   Args: {call['args']}")
    for message in (step.get("tools") or {}).get("messages", []):
        print(f"✅ Tool Result ({message.name}): {str(message.content)[:500]}\n---")


def _delegate_tool(agent, name: str, description: str):
    """Wrap a sub-agent as a tool that the supervisor can call.

    Arguments:
    agent -- compiled agent returned by create_agent
    name -- tool name shown to the supervisor model
    description -- tells the supervisor when to delegate to this agent

    Returns a LangChain tool taking a single "request" string.
    """

    @tool(name, description=description)
    def delegate(request: str) -> str:
        """Pass a request to the sub-agent and return its final answer."""
        answer = ""
        for step in agent.stream({"messages": [{"role": "user", "content": request}]}):
            _show_step(step)
            for message in (step.get("model") or {}).get("messages", []):
                answer = message.text or answer
        return answer or "The agent returned no answer."

    return delegate


def build_supervisor(database: str | None = None):
    """Build the supervisor agent and the sub-agents it delegates to.

    Keyword arguments:
    database -- path to a SQLite file; when given, a database_agent with
                read-only SQL tools is added

    Returns a compiled agent with in-memory conversation history. Pass a
    thread_id in the invoke config to keep context between turns.
    """
    llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL"), temperature=0)

    research_agent = create_agent(
        llm,
        tools=[
            DuckDuckGoSearchRun(),
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2)),
        ],
        system_prompt=(
            "You research topics on the web. Use Wikipedia for background and "
            "DuckDuckGo for current information. Report facts concisely and "
            "say which tool each fact came from."
        ),
    )
    security_agent = create_agent(
        llm,
        tools=[cve_lookup, cve_search, dns_lookup],
        system_prompt=(
            "You are a security analyst. Use the NVD tools for vulnerability "
            "data and the DNS tool for domain records. Report only what the "
            "tools return; never invent CVE identifiers or scores."
        ),
    )
    coding_agent = create_agent(
        llm,
        tools=[run_python],
        system_prompt=(
            "You solve problems by writing and running Python. Always print "
            "results. If the user declines to run code, say so and stop."
        ),
    )
    delegates = [
        _delegate_tool(
            research_agent,
            "research_agent",
            "Delegate web and encyclopedia research. Give a complete, "
            "self-contained request.",
        ),
        _delegate_tool(
            security_agent,
            "security_agent",
            "Delegate CVE lookups, CVE keyword searches, and DNS record "
            "lookups. Give a complete, self-contained request.",
        ),
        _delegate_tool(
            coding_agent,
            "coding_agent",
            "Delegate calculations and data processing that need Python. "
            "Include all data the code needs in the request.",
        ),
    ]

    if database:
        db = SQLDatabase.from_uri(
            f"sqlite:///file:{os.path.abspath(database)}?mode=ro&uri=true",
            sample_rows_in_table_info=1,
            max_string_length=200,
        )
        database_agent = create_agent(
            llm,
            tools=SQLDatabaseToolkit(db=db, llm=llm).get_tools(),
            system_prompt=(
                "You answer questions about a SQLite database. List the "
                "tables first, read the schema of only the tables you need, "
                "then run SELECT queries that name specific columns and use "
                "LIMIT. Never modify data. Answer only from query results."
            ),
        )
        delegates.append(
            _delegate_tool(
                database_agent,
                "database_agent",
                f"Delegate questions about the loaded database "
                f"({os.path.basename(database)}), such as its users, roles, "
                f"and applications. Give a complete, self-contained request.",
            )
        )

    return create_agent(
        llm,
        tools=delegates,
        system_prompt=(
            "You are SecOps Sidekick, a supervisor that answers security and "
            "IT questions by delegating to specialist agents. Break the "
            "question into steps, call the right agent for each step, pass "
            "earlier results forward when a later step needs them, then write "
            "one clear final answer. The agents cannot see this conversation."
        ),
        checkpointer=InMemorySaver(),
    )


def ask(supervisor, message: str, thread_id: str = "default") -> str:
    """Send one user message to the supervisor and return its reply text.

    Tool calls made along the way are printed as they happen.

    Arguments:
    supervisor -- agent returned by build_supervisor
    message -- the user's question

    Keyword arguments:
    thread_id -- conversation identifier; reuse it to keep chat history
    """
    answer = ""
    for step in supervisor.stream(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},
    ):
        _show_step(step)
        for reply in (step.get("model") or {}).get("messages", []):
            answer = reply.text or answer
    return answer


def main() -> None:
    """Run SecOps Sidekick from the command line.

    With a question argument, answer it and exit. Without one, start a chat
    loop that ends on a blank line, "exit", "quit", or end-of-file.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--db")
    parser.add_argument("question", nargs="*")
    args = parser.parse_args()
    if args.help:
        print(__doc__)
        return
    for name in ("GOOGLE_API_KEY", "GOOGLE_MODEL"):
        if not os.environ.get(name):
            sys.exit(f"Set the {name} environment variable first (see -h).")
    if args.db and not os.path.isfile(args.db):
        sys.exit(f"Database file not found: {args.db}")

    supervisor = build_supervisor(args.db)
    if args.question:
        print(ask(supervisor, " ".join(args.question)))
        return

    print("SecOps Sidekick -- ask a question; a blank line exits.")
    while True:
        try:
            message = input("\nllm>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message or message.lower() in ("exit", "quit"):
            break
        try:
            print(f"\n{ask(supervisor, message)}")
        except Exception as error:  # keep the session alive on API errors
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
