"""
An original 200-question corpus meant to look like real assistant traffic.

Why not MMLU / TriviaQA / HotpotQA
----------------------------------
They were convenient, not representative. MMLU is 4-way multiple choice — nobody
asks an assistant to pick A/B/C/D. TriviaQA is deliberately obscure pub trivia.
HotpotQA's questions are synthetically welded 2-hop constructions ("the university
where X was a professor") that read nothing like what a person types. Routing
decisions tuned on that traffic are tuned on the wrong distribution.

What this is instead
--------------------
Questions written to mirror the mix an assistant actually sees: quick lookups,
explanations, coding and debugging, comparisons and recommendations, how-tos,
calculations, and genuinely multi-part requests.

Grading
-------
Every item carries `must_include`: a list of requirements, each either
  - a string      -> that term must appear, or
  - a list        -> at least one of those alternatives must appear
Matching is case-insensitive over normalised text. This keeps grading deterministic
and free — no LLM judge, whose cost this project already showed cannot be justified.

Assertions are deliberately loose enough to accept any correct phrasing and strict
enough to reject a wrong or evasive answer. Where a question has no single defensible
answer it is marked gradeable=False and contributes to cost/routing stats only.

`expected_path` is the tier that *should* handle the item, and is the ground-truth
label for routing accuracy:
    cheap      one fact, no reasoning — the "I'd have googled it" tier
    middle     explanation, code, analysis — the default
    decompose  several genuinely separable sub-questions in one request
"""

# ---------------------------------------------------------------------------
# LOOKUPS — single fact, no reasoning. Should route to `cheap`.
# ---------------------------------------------------------------------------

LOOKUPS = [
    ("What's the capital of Australia?", ["canberra"]),
    ("Who wrote Pride and Prejudice?", ["austen"]),
    ("What year did the Berlin Wall fall?", ["1989"]),
    ("How many bones are in the adult human body?", ["206"]),
    ("What's the chemical symbol for gold?", [["au", "aurum"]]),
    ("Who painted the Mona Lisa?", [["leonardo", "da vinci", "vinci"]]),
    ("What's the largest ocean on Earth?", ["pacific"]),
    ("What language do they speak in Brazil?", ["portuguese"]),
    ("How many players from one team are on a soccer field at once?", [["11", "eleven"]]),
    ("What's the boiling point of water in Fahrenheit?", ["212"]),
    ("Who was the first person to walk on the moon?", ["armstrong"]),
    ("What's the currency of Japan?", ["yen"]),
    ("What's the tallest mountain in the world?", ["everest"]),
    ("Who came up with the theory of general relativity?", ["einstein"]),
    ("What does HTTP stand for?", ["hypertext", "transfer", "protocol"]),
    ("What's the smallest prime number?", [["2", "two"]]),
    ("In what year did World War II end?", ["1945"]),
    ("What's the main ingredient in guacamole?", ["avocado"]),
    ("Who wrote the novel 1984?", ["orwell"]),
    ("Which planet is known as the Red Planet?", ["mars"]),
    ("What's the largest mammal?", [["blue whale", "blue-whale"]]),
    ("Who composed the symphony containing Ode to Joy?", ["beethoven"]),
    ("What's the capital of Canada?", ["ottawa"]),
    ("How many sides does a hexagon have?", [["6", "six"]]),
    ("Which gas do plants absorb from the air?", [["carbon dioxide", "co2"]]),
    ("Who invented the telephone?", ["bell"]),
    ("What's the freezing point of water in Celsius?", [["0", "zero"]]),
    ("What does DNA stand for?", ["deoxyribonucleic"]),
    ("What's the hardest naturally occurring substance?", ["diamond"]),
    ("What's the currency of the United Kingdom?", [["pound", "sterling", "gbp"]]),
    ("Who directed Jurassic Park?", ["spielberg"]),
    ("What's the atomic number of carbon?", [["6", "six"]]),
    ("Which city is the Colosseum in?", ["rome"]),
    ("Who wrote Romeo and Juliet?", ["shakespeare"]),
    ("What's the chemical symbol for iron?", ["fe"]),
    ("What port does HTTPS use by default?", ["443"]),
    ("What does API stand for?", ["application", "programming", "interface"]),
    ("Which Unix command lists the files in a directory?", ["ls"]),
    ("What does CSS stand for?", ["cascading", "style", "sheets"]),
    ("What's the default port for PostgreSQL?", ["5432"]),
    ("What does JSON stand for?", ["javascript", "object", "notation"]),
    ("Who created the Python programming language?", [["guido", "rossum"]]),
    ("What does SQL stand for?", [["structured query language", "structured"]]),
    ("What does RAM stand for?", ["random access memory"]),
    ("What's the default port for SSH?", ["22"]),
    ("What HTTP status code means Not Found?", ["404"]),
    ("What does CPU stand for?", ["central processing unit"]),
    ("Which planet is closest to the sun?", ["mercury"]),
    ("What's the capital of Switzerland?", ["bern"]),
    ("How many minutes are in a full day?", ["1440"]),
]

# ---------------------------------------------------------------------------
# EXPLANATIONS — conceptual, needs real prose. Should route to `middle`.
# ---------------------------------------------------------------------------

EXPLANATIONS = [
    ("Explain the difference between TCP and UDP.",
     ["tcp", "udp", ["connection", "handshake"], ["reliab", "ordered", "guarantee"]]),
    ("Why is the sky blue?",
     [["scatter", "rayleigh"], ["wavelength", "shorter", "blue"]]),
    ("How does a refrigerator actually keep things cold?",
     [["compress", "compressor"], ["refrigerant", "evaporat", "coolant"], "heat"]),
    ("What causes inflation in an economy?",
     [["money supply", "demand", "printing"], ["price", "prices"]]),
    ("Explain photosynthesis in simple terms.",
     ["light", ["carbon dioxide", "co2"], ["glucose", "sugar", "energy"], "oxygen"]),
    ("What's the difference between weather and climate?",
     ["weather", "climate", ["long", "average", "decades"]]),
    ("How do vaccines work?",
     ["immune", ["antibod", "immunit", "response"], ["pathogen", "virus", "antigen"]]),
    ("Why do we have seasons?",
     [["tilt", "axis"], ["orbit", "sun", "revolution"]]),
    ("Explain how compound interest works.",
     ["interest", ["principal", "balance"], ["compound", "exponential", "reinvest"]]),
    ("What's the difference between RAM and disk storage?",
     ["ram", [ "volatile", "temporary", "power"], ["disk", "storage", "persistent"]]),
    ("Why does a boat float but a nail sinks?",
     [["density", "displace", "buoyan"]]),
    ("Explain what a black hole is.",
     [["gravity", "gravitational"], ["light", "escape"], ["mass", "collapse", "dense"]]),
    ("How does encryption keep my messages private?",
     [["key", "cipher"], ["encrypt", "decrypt"]]),
    ("What's the greenhouse effect?",
     [["heat", "infrared", "radiation"], ["trap", "absorb"], ["atmosphere", "gases"]]),
    ("Why do onions make you cry?",
     [["sulf", "syn-propanethial", "enzyme", "gas"], ["eye", "tear"]]),
    ("Explain the difference between a virus and a bacterium.",
     ["virus", [ "bacteri"], ["host", "cell", "living", "reproduce"]]),
    ("How does GPS know where I am?",
     [["satellite", "satellites"], ["signal", "time", "triangulat", "trilaterat"]]),
    ("What is machine learning, in plain English?",
     [["data", "examples"], ["pattern", "learn", "train"], ["predict", "model"]]),
    ("Why is the ocean salty?",
     [["rock", "erosion", "rivers", "weathering"], ["salt", "mineral", "sodium"]]),
    ("Explain what inflation targeting means for a central bank.",
     [["central bank", "policy"], ["target", "2%", "rate"], ["inflation"]]),
    ("How do noise-cancelling headphones work?",
     [["microphone", "mic"], ["opposite", "inverted", "anti-phase", "out of phase", "destructive"]]),
    ("What's the difference between a recession and a depression?",
     ["recession", "depression", ["severe", "longer", "deeper", "duration"]]),
    ("Why do airplanes leave white trails in the sky?",
     [["contrail", "condens", "water vapor", "vapour"], ["ice", "cold", "crystal"]]),
    ("Explain what an API rate limit is and why services use them.",
     [["rate limit", "throttl"], ["abuse", "overload", "fair", "capacity", "protect"]]),
    ("How does a nuclear reactor generate electricity?",
     [["fission", "split"], ["heat", "steam"], ["turbine", "generator"]]),
    ("What's the difference between HTTP and HTTPS?",
     ["https", ["encrypt", "tls", "ssl", "secure"]]),
    ("Why do we dream?",
     [["memory", "consolidat", "process", "rem", "theory"]]),
    ("Explain what technical debt means in software.",
     [["shortcut", "quick", "expedient", "trade"], ["later", "interest", "cost", "refactor", "maintain"]]),
    ("How does a search engine decide what to show first?",
     [["rank", "relevance", "algorithm"], ["link", "keyword", "quality", "signal"]]),
    ("What's the difference between weather forecasting and climate modelling?",
     ["forecast", ["climate"], ["short", "long", "average", "trend"]]),
    ("Explain what happens when you type a URL into a browser and hit enter.",
     [["dns", "resolve"], ["request", "http", "tcp"], ["render", "response", "html"]]),
    ("Why does bread rise?",
     [["yeast", "leaven", "baking"], ["carbon dioxide", "co2", "gas"]]),
    ("What is herd immunity?",
     [["immune", "vaccinat", "resistant"], ["spread", "transmission", "protect"]]),
    ("Explain the difference between correlation and causation.",
     ["correlation", ["causation", "cause"], ["not", "does not", "doesn't"]]),
    ("How do solar panels turn sunlight into electricity?",
     [["photovoltaic", "semiconductor", "silicon"], ["electron", "current", "photon"]]),
]

# ---------------------------------------------------------------------------
# TECH / CODING / DEBUGGING — should route to `middle`.
# ---------------------------------------------------------------------------

TECH = [
    ("Write a Python function that reverses a string.",
     ["def", [ "[::-1]", "reversed", "reverse"]]),
    ("How do I undo the last git commit but keep my changes?",
     [["reset", "soft"], ["head~", "head^", "head~1"]]),
    ("What's the difference between `==` and `===` in JavaScript?",
     [["type", "coerc", "conver"], ["strict"]]),
    ("My Python script says 'ModuleNotFoundError: No module named requests'. How do I fix it?",
     [["pip install", "install", "uv add"], ["requests"]]),
    ("Write a SQL query that finds the 5 highest-paid employees.",
     ["select", "order by", ["limit", "top", "fetch first"]]),
    ("What does the `async` keyword do in Python?",
     [["coroutine", "asynchronous", "await"], ["event loop", "concurrent", "non-blocking", "await"]]),
    ("How do I check which process is using port 8080 on Linux?",
     [["lsof", "netstat", "ss ", "fuser"]]),
    ("Explain what a Docker container is versus a virtual machine.",
     ["container", [ "kernel", "os", "operating system"], ["virtual machine", "vm", "hypervisor"]]),
    ("My React component re-renders on every keystroke and it's slow. What should I look at?",
     [["memo", "usememo", "usecallback", "debounce", "state"]]),
    ("What's a database index and when should I add one?",
     [["index"], ["lookup", "query", "speed", "faster", "scan"], ["write", "insert", "space", "cost", "slow"]]),
    ("Write a bash one-liner that counts lines in all .py files in a directory.",
     [["wc -l", "wc"], [".py", "*.py"]]),
    ("What's the difference between `git merge` and `git rebase`?",
     ["merge", "rebase", ["history", "linear", "commit"]]),
    ("How do I make an HTTP POST request in Python?",
     [["requests.post", "httpx", "urllib", "post"]]),
    ("Explain what a race condition is and give an example.",
     [["concurren", "thread", "timing", "order", "simultaneous"], ["shared", "state", "resource", "variable"]]),
    ("What does HTTP 429 mean and how should a client handle it?",
     [["too many", "rate limit"], ["retry", "back off", "backoff", "wait"]]),
    ("How do I find and remove duplicate rows in a pandas DataFrame?",
     [["duplicated", "drop_duplicates"]]),
    ("What's the difference between a list and a tuple in Python?",
     [["mutable", "immutable", "change"]]),
    ("My Docker build is really slow. What are the usual causes?",
     [["cache", "layer", "context", "order"]]),
    ("Explain what CORS is and why my browser is blocking my API call.",
     [["cross-origin", "cross origin", "cors"], ["header", "access-control", "origin"]]),
    ("Write a regex that matches a valid email address.",
     [["@"], ["[", "\\\\w", "a-z"]]),
    ("What's the difference between authentication and authorization?",
     [["authentic"], ["authoriz"], ["who", "identity"], ["permission", "allowed", "access", "what"]]),
    ("How does garbage collection work in a language like Java or Python?",
     [["reference", "reachable", "unreachable", "refcount", "mark"], ["memory", "free", "reclaim"]]),
    ("My SQL query is slow. What's the first thing I should check?",
     [["explain", "index", "plan", "analyze"]]),
    ("What's the difference between REST and GraphQL?",
     ["rest", "graphql", ["endpoint", "query", "over-fetch", "overfetch", "single"]]),
    ("How do I set an environment variable in a Dockerfile?",
     [["env "]]),
    ("Explain what a memory leak is and how to spot one.",
     [["memory", "heap"], ["grow", "release", "free", "not freed", "leak"]]),
    ("Write a Python function that checks whether a string is a palindrome.",
     ["def", [ "[::-1]", "reversed", "reverse"]]),
    ("What's the difference between a process and a thread?",
     ["process", "thread", ["memory", "address space", "share"]]),
    ("How do I revert a file to the version in the last commit with git?",
     [["checkout", "restore"]]),
    ("What is idempotency in an API and why does it matter?",
     [["idempot"], ["same", "repeat", "retry", "twice", "once"]]),
    ("Explain the difference between synchronous and asynchronous code.",
     [["block", "wait", "synchronous"], ["asynchronous", "concurrent", "non-blocking", "continue"]]),
    ("How do I profile a slow Python function?",
     [["cprofile", "profile", "timeit", "line_profiler", "py-spy"]]),
    ("What's a JWT and how is it used for auth?",
     [["json web token", "jwt"], ["sign", "claim", "payload", "token", "header"]]),
    ("Write a Python list comprehension that squares the even numbers from 1 to 20.",
     [["for", "range"], ["%", "even", "2"], ["**", "*"]]),
    ("What's the difference between horizontal and vertical scaling?",
     [["horizontal", "scale out", "more machines", "more servers"], ["vertical", "scale up", "bigger"]]),
]

# ---------------------------------------------------------------------------
# ANALYSIS / COMPARISON / ADVICE — judgement, one coherent answer. `middle`.
# ---------------------------------------------------------------------------

ANALYSIS = [
    ("Should a small startup use microservices or a monolith to begin with?",
     [["monolith"], ["microservice"], ["complex", "overhead", "small", "start", "team"]]),
    ("What are the main trade-offs between renting and buying a home?",
     [["rent"], ["buy", "mortgage", "own"], ["flexib", "equity", "maintenance", "upfront", "deposit"]]),
    ("Compare Postgres and MongoDB for an app with lots of relational data.",
     ["postgres", ["mongo"], ["relational", "join", "schema", "document"]]),
    ("Is it better to pay off debt or invest spare cash? Explain the reasoning.",
     [["interest rate", "rate", "return"], ["debt"], ["invest"]]),
    ("What are the pros and cons of remote work for a small engineering team?",
     [["remote"], ["communicat", "collaborat", "onboard", "culture", "async"], ["flexib", "commute", "talent", "cost"]]),
    ("Why might a company choose to stay private instead of going public?",
     [["public", "ipo"], ["control", "disclosure", "reporting", "scrutiny", "short-term", "regulat"]]),
    ("Compare electric cars and petrol cars on total cost of ownership.",
     [["electric", "ev"], ["fuel", "petrol", "gas", "charging"], ["maintenance", "depreciat", "upfront", "purchase"]]),
    ("What should I consider when choosing between AWS, GCP and Azure?",
     [["aws"], ["gcp", "google"], ["azure"], ["cost", "pricing", "service", "lock-in", "ecosystem", "team"]]),
    ("Explain the trade-offs of using an ORM versus writing raw SQL.",
     [["orm"], ["sql"], ["control", "performance", "productiv", "abstract", "n+1"]]),
    ("Is nuclear power a good way to cut carbon emissions? Give both sides.",
     [["carbon", "emission"], ["waste", "cost", "safety", "risk", "time", "accident"], ["baseload", "reliable", "low-carbon", "density"]]),
    ("What are the risks of relying on a single cloud provider?",
     [["lock-in", "lock in", "dependen"], ["outage", "downtime", "price", "leverage", "migrat"]]),
    ("Compare TypeScript and JavaScript for a team of five engineers.",
     ["typescript", ["javascript"], ["type", "safety", "refactor", "tooling", "build", "overhead"]]),
    ("How should a small team decide what to build next?",
     [["user", "customer", "feedback", "data", "impact"], ["effort", "cost", "priorit", "value"]]),
    ("What are the downsides of over-indexing a database table?",
     [["write", "insert", "update"], ["space", "storage", "slow", "overhead", "maintain"]]),
    ("Explain why premature optimisation is considered a problem.",
     [["premature"], ["measure", "profile", "bottleneck", "complexity", "readab", "wrong"]]),
    ("Compare renting servers versus serverless for a bursty workload.",
     [["serverless", "lambda", "function"], ["idle", "burst", "scale", "cold start", "cost"]]),
    ("What makes a good code review?",
     [["specific", "actionable", "small", "clear", "kind", "constructive"], ["logic", "design", "correct", "readab", "test"]]),
    ("Why do most software projects run over schedule?",
     [["estimat", "unknown", "scope", "complexity", "optimis", "optimiz"]]),
    ("What are the arguments for and against a four-day work week?",
     [["productiv", "burnout", "wellbeing", "retention"], ["cost", "coverage", "customer", "workload", "cram"]]),
    ("Should I learn Rust or Go for backend work? Explain the trade-offs.",
     ["rust", "go", ["learning curve", "simpl", "memory", "performance", "concurren", "ecosystem"]]),
    ("What are the main causes of burnout in engineering teams?",
     [["workload", "hours", "pressure", "deadline"], ["control", "autonomy", "recognition", "meaning", "unclear"]]),
    ("Explain the trade-offs between strong consistency and eventual consistency.",
     [["consisten"], ["availab", "latency", "partition", "cap", "stale"]]),
    ("How would you evaluate whether a new feature was successful?",
     [["metric", "measure", "kpi"], ["baseline", "before", "a/b", "experiment", "usage", "retention"]]),
    ("What are the risks of using an LLM in a customer-facing product?",
     [["hallucinat", "wrong", "inaccurat", "error"], ["cost", "latency", "privacy", "prompt injection", "safety", "brand"]]),
    ("Compare batch processing and stream processing.",
     [["batch"], ["stream", "real-time", "realtime"], ["latency", "throughput", "window"]]),
]

# ---------------------------------------------------------------------------
# PRACTICAL / HOW-TO / CALCULATION — `middle`.
# ---------------------------------------------------------------------------

PRACTICAL = [
    ("If I invest $5,000 at 6% annual interest compounded yearly, what's it worth in 10 years?",
     [["8954", "8,954", "8955", "8,955", "8900", "8,9"]]),
    ("A recipe serves 4 and needs 300g of flour. How much flour for 7 people?",
     [["525"]]),
    ("I earn $85,000 a year. What's that per month before tax?",
     [["7083", "7,083", "7,08"]]),
    ("How long does it take to drive 420 km at an average of 70 km/h?",
     [["6 hour", "6 hours", "six hour"]]),
    ("Convert 98.6 degrees Fahrenheit to Celsius.",
     [["37"]]),
    ("If a shirt is $80 with 25% off, what do I pay?",
     [["60"]]),
    ("My server handles 300 requests per second. How many is that per day?",
     [["25,920,000", "25920000", "25.9", "2.592"]]),
    ("What's 15% of 2,400?",
     [["360"]]),
    ("How do I make cold brew coffee at home?",
     [["coarse", "ground", "grind"], ["12", "18", "24", "hour", "overnight", "fridge"], ["water"]]),
    ("How do I get a red wine stain out of a white shirt?",
     [["cold water", "salt", "baking soda", "hydrogen peroxide", "club soda", "blot"]]),
    ("What's a good way to structure a 30-minute daily workout with no equipment?",
     [["warm", "squat", "push", "plank", "lunge", "burpee", "rest"]]),
    ("How should I prepare for a technical interview in two weeks?",
     [["practice", "problem", "leetcode", "mock"], ["review", "fundamental", "data structure", "algorithm", "system design"]]),
    ("How do I back up my photos so I don't lose them?",
     [["cloud", "external", "drive", "backup"], ["3-2-1", "two", "multiple", "offsite", "copy", "second"]]),
    ("What's a reasonable way to split rent when bedrooms are different sizes?",
     [["size", "square", "area", "proportion", "ratio"]]),
    ("How do I read a nutrition label properly?",
     [["serving"], ["calorie", "sugar", "sodium", "fat", "ingredient", "daily value"]]),
]

# ---------------------------------------------------------------------------
# MULTI-PART — several genuinely separable sub-questions. `decompose`.
# ---------------------------------------------------------------------------

MULTIPART = [
    ("I'm building a mobile app for 10,000 users. What database should I pick, where should I host it, and roughly what will it cost per month?",
     [["postgres", "database", "dynamo", "firebase", "sql"], ["host", "aws", "gcp", "vercel", "railway", "fly", "render"], ["$", "cost", "month"]]),
    ("Compare Postgres and MySQL for a write-heavy workload, then tell me which you'd pick and why.",
     ["postgres", "mysql", ["write", "insert", "throughput"], ["recommend", "pick", "choose", "i'd", "would"]]),
    ("Explain what Kubernetes does, then tell me whether a 3-person startup should use it.",
     [["kubernetes", "orchestrat", "container"], ["no", "not", "overkill", "probably", "unless", "complexity"]]),
    ("What were the main causes of the 2008 financial crisis, and what regulations came out of it?",
     [["subprime", "mortgage", "housing", "derivative", "leverage"], ["dodd-frank", "basel", "regulat", "stress test", "capital requirement"]]),
    ("Summarise how HTTPS works, then explain what a certificate authority does in that process.",
     [["tls", "ssl", "encrypt", "handshake"], ["certificate authority", "ca", "sign", "trust", "verif"]]),
    ("I want to learn data science. What should I learn first, what resources should I use, and how long will it take?",
     [["python", "statistic", "sql", "math"], ["course", "book", "kaggle", "project", "resource"], ["month", "year", "week", "time"]]),
    ("Explain the difference between supervised and unsupervised learning, and give an example of each.",
     [["supervised"], ["unsupervised"], ["label"], ["cluster", "classif", "regression", "example"]]),
    ("What is inflation, what causes it, and what can a central bank do about it?",
     [["price", "purchasing power"], ["demand", "supply", "money", "cost"], ["interest rate", "rate", "tighten", "policy"]]),
    ("Compare React, Vue and Svelte, then recommend one for a team new to frontend.",
     ["react", "vue", "svelte", ["recommend", "pick", "choose", "would", "suggest"]]),
    ("Explain what technical debt is, how to measure it, and how to make time to pay it down.",
     [["shortcut", "trade", "quick", "expedient"], ["measure", "metric", "track", "time", "bug", "velocity"], ["allocat", "budget", "sprint", "percent", "20%", "time"]]),
    ("What's the difference between a 401k and an IRA, and which should I prioritise?",
     [["401", "employer"], ["ira"], ["match", "priorit", "first", "limit", "tax"]]),
    ("Walk me through setting up CI for a Python project: what tool, what steps, and what should it run?",
     [["github actions", "gitlab", "circle", "ci"], ["test", "pytest"], ["lint", "install", "step", "workflow", "yaml"]]),
    ("Explain what caching is, describe two caching strategies, and tell me when caching hurts.",
     [["cache", "cach"], ["ttl", "lru", "write-through", "write-back", "invalidat", "aside"], ["stale", "invalidat", "wrong", "memory", "hurt"]]),
    ("What caused the fall of the Roman Empire, and which of those causes do historians disagree about?",
     [["economic", "military", "invasion", "political", "barbarian", "division"], ["debate", "disagree", "contest", "argue", "dispute", "no consensus"]]),
    ("I have a slow API endpoint. Tell me how to diagnose it, what the common causes are, and how to fix each.",
     [["profil", "measure", "trace", "log", "monitor"], ["database", "query", "n+1", "network", "serial", "blocking"], ["index", "cache", "batch", "async", "optimi"]]),
    ("Explain how a bill becomes law in the US, then describe two ways it can fail along the way.",
     [["congress", "house", "senate"], ["president", "sign", "veto"], ["filibuster", "committee", "veto", "die", "fail", "adjourn"]]),
    ("Compare the health effects of running and swimming, then say which is better for someone with knee problems.",
     ["running", "swimming", ["impact", "joint", "knee"], ["swim"]]),
    ("What is quantum computing, what problems could it actually solve, and how far away is it?",
     [["qubit", "superposition", "quantum"], ["factor", "simulat", "optimis", "optimiz", "crypt", "chemistry"], ["year", "decade", "away", "early", "far"]]),
    ("Explain the water cycle, then describe how climate change is altering it.",
     [["evaporat", "condens", "precipitat"], ["climate", "warming"], ["intens", "extreme", "drought", "flood", "change"]]),
    ("Design a simple URL shortener: what's the data model, how do you generate short codes, and how would you scale it?",
     [["table", "schema", "key", "id", "url"], ["hash", "base62", "counter", "random", "encode"], ["cache", "shard", "replica", "scale", "load"]]),
    ("What are the main renewable energy sources, what are their drawbacks, and which is growing fastest?",
     [["solar", "wind", "hydro"], ["intermitten", "storage", "land", "cost", "variab"], ["solar", "wind", "fastest", "growing"]]),
    ("Explain what an index does in a database, when to add one, and when it makes things worse.",
     [["index", "lookup", "scan"], ["query", "where", "join", "filter", "frequent"], ["write", "insert", "update", "space", "overhead"]]),
    ("I'm moving to a new city for work. What should I research about the area, in what order, and what's easy to overlook?",
     [["rent", "cost", "housing", "neighbourhood", "neighborhood"], ["commute", "transport", "school", "safety"], ["overlook", "forget", "easy to miss", "often", "hidden"]]),
    ("Explain the difference between machine learning and deep learning, and give a case where deep learning is overkill.",
     [["machine learning"], ["deep learning", "neural"], ["overkill", "simpler", "small", "tabular", "regression", "logistic"]]),
    ("What is the placebo effect, why does it complicate drug trials, and how do researchers control for it?",
     [["placebo"], ["trial", "complicat", "confound", "bias"], ["double-blind", "double blind", "control group", "randomi"]]),
    ("Summarise what containers do, compare Docker with Podman, and say when you'd choose each.",
     [["container", "isolat"], ["docker"], ["podman"], ["rootless", "daemon", "choose", "prefer", "when"]]),
    ("Explain what an LLM context window is, why it's limited, and what to do when your input is too big.",
     [["context window", "token"], ["memory", "attention", "quadratic", "compute", "cost", "limit"], ["chunk", "summar", "rag", "retriev", "split"]]),
    ("What causes traffic jams even when there's no accident, and what can cities do about them?",
     [["phantom", "wave", "density", "braking", "reaction", "capacity"], ["transit", "pricing", "congestion charge", "lane", "public transport", "ramp"]]),
    ("Compare SQL and NoSQL databases, then tell me what questions I should ask before choosing.",
     ["sql", ["nosql", "document", "key-value"], ["schema", "join", "scale", "consisten"], ["question", "ask", "consider", "depend"]]),
    ("Explain how vaccines are developed and tested, and why it normally takes years.",
     [["phase", "trial", "clinical"], ["safety", "efficacy", "volunteer", "participant"], ["year", "time", "long", "recruit", "regulat", "approval"]]),
]

# ---------------------------------------------------------------------------
# NEWS / CURRENT AFFAIRS — the "what's the deal with X" register. `middle`.
#
# Deliberately anchored on established background rather than breaking events:
# a benchmark whose answers change weekly cannot be graded deterministically, and
# an ungradeable question measures nothing but cost.
# ---------------------------------------------------------------------------

NEWS = [
    ("What is the CHIPS Act and what was it meant to achieve?",
     [["semiconductor", "chip"], ["manufactur", "domestic", "united states", "us ", "subsid", "onshore"]]),
    ("What does OPEC do and why do its decisions move oil prices?",
     [["oil", "petroleum"], ["production", "output", "quota", "supply"], ["price"]]),
    ("What is the EU's GDPR and who does it apply to?",
     [["data protection", "privacy", "personal data"], ["eu", "european"], ["resident", "citizen", "process", "compan", "anyone"]]),
    ("Why do central banks raise interest rates to fight inflation?",
     [["borrow", "cost", "demand", "spend"], ["cool", "slow", "reduce", "lower"], ["inflation", "price"]]),
    ("What is the Paris Climate Agreement and what does it commit countries to?",
     [["paris"], ["emission", "warming", "temperature"], ["1.5", "2 degree", "two degree", "target", "pledge", "nationally determined"]]),
    ("What is quantitative easing and when do central banks use it?",
     [["buy", "purchase", "asset", "bond"], ["money supply", "liquidity", "stimul", "rate"], ["recession", "crisis", "downturn", "low"]]),
    ("What does it mean when a country's currency is described as a reserve currency?",
     [["reserve"], ["central bank", "hold", "foreign exchange", "trade", "settle"], ["dollar", "usd", "stable", "demand"]]),
    ("What is the WHO and what powers does it actually have?",
     [["world health organization", "world health organisation", "who"], ["guidance", "recommend", "coordinat", "advis"], ["cannot", "no ", "not", "limited", "sovereign", "member state"]]),
    ("Explain what a trade deficit is and whether it's necessarily bad.",
     [["import", "export"], ["deficit"], ["not necessarily", "not always", "depends", "not inherently", "can be"]]),
    ("What are semiconductors used for and why are they geopolitically important?",
     [["chip", "semiconductor", "processor"], ["electronic", "device", "computer", "phone", "car", "military"], ["supply chain", "taiwan", "concentrat", "strategic", "depend"]]),
]

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

CATEGORIES = [
    ("lookup", "cheap", LOOKUPS),
    ("explanation", "middle", EXPLANATIONS),
    ("tech", "middle", TECH),
    ("analysis", "middle", ANALYSIS),
    ("practical", "middle", PRACTICAL),
    ("news", "middle", NEWS),
    ("multipart", "decompose", MULTIPART),
]


def build() -> list[dict]:
    rows, n = [], 0
    for category, expected_path, items in CATEGORIES:
        for prompt, must_include in items:
            n += 1
            rows.append({
                "source_id": f"syn:{n:03d}",
                "prompt": prompt,
                "category": category,
                "expected_path": expected_path,
                "must_include": must_include,
                "gradeable": True,
            })
    return rows


if __name__ == "__main__":
    import collections
    import json
    from pathlib import Path

    rows = build()
    out = Path(__file__).parent / "synthetic_200.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} questions -> {out}")
    print("  by category:   ", dict(collections.Counter(r["category"] for r in rows)))
    print("  by expected:   ", dict(collections.Counter(r["expected_path"] for r in rows)))

    # A duplicate prompt is a silent sampling bug, and an empty assertion list makes
    # an item unconditionally correct — both would quietly inflate accuracy.
    prompts = [r["prompt"] for r in rows]
    assert len(set(prompts)) == len(prompts), "duplicate prompts"
    assert all(r["must_include"] for r in rows), "an item has no assertions"
    print("  checks passed: no duplicates, every item has assertions")
