"""
Sample text fixtures for testing AI detection.

These texts are used to generate test PDFs with known content types
for verification of the AI detection and colorization pipeline.
"""

# Human-written text (more varied, less predictable, natural errors/quirks)
HUMAN_TEXT_SHORT = """
I've been thinking about the way sunlight filters through leaves on a lazy
afternoon. There's something almost magical about it - the dappled patterns
shifting on the ground, each movement unique and unpredictable. My dog doesn't
seem to care much, though. He's more interested in the squirrel that just
darted up the oak tree. Classic.
"""

HUMAN_TEXT_LONG = """
Last Tuesday, I attempted to make my grandmother's famous apple pie recipe,
which she'd scribbled on a coffee-stained index card sometime in the 1970s.
The instructions were characteristically vague: "add enough butter," "bake
until done," and my personal favorite, "you'll know when it's ready." Well,
I didn't know. My first attempt came out looking like a geological experiment
gone wrong - all crusty peaks and molten valleys.

But here's the thing about family recipes: they're not really about precision.
They're about the memories that come flooding back when you smell cinnamon and
butter mingling in the oven. They're about the laughter when things go wrong,
and the quiet satisfaction when, on the third try, you pull out something that
actually resembles what you remember from childhood.

My grandmother used to say that cooking is one part science, two parts intuition,
and three parts love. I'm not sure the math checks out, but I think I finally
understand what she meant.
"""

# AI-generated text (more structured, predictable patterns, formal)
AI_TEXT_SHORT = """
Artificial intelligence represents a significant advancement in modern technology.
Machine learning algorithms can process vast amounts of data to identify patterns
and make predictions. These systems have applications across numerous industries,
including healthcare, finance, and transportation. As AI continues to evolve, it
is important to consider both its potential benefits and ethical implications.
"""

AI_TEXT_LONG = """
The development of renewable energy sources has become increasingly important
in addressing climate change and reducing dependence on fossil fuels. Solar
and wind power technologies have made significant progress in recent years,
becoming more efficient and cost-effective. Governments and private organizations
are investing heavily in these sustainable energy solutions.

Renewable energy offers numerous advantages over traditional energy sources.
First, it produces minimal greenhouse gas emissions, helping to mitigate climate
change. Second, renewable resources are essentially inexhaustible, unlike finite
fossil fuel reserves. Third, the technology creates new employment opportunities
in manufacturing, installation, and maintenance sectors.

However, challenges remain in the widespread adoption of renewable energy.
Energy storage solutions need further development to address the intermittent
nature of solar and wind power. Infrastructure improvements are necessary to
support distributed energy generation. Additionally, initial investment costs
can be substantial, though long-term savings often offset these expenses.

Looking forward, the transition to renewable energy appears inevitable.
Technological innovations continue to improve efficiency and reduce costs.
Policy frameworks are evolving to support sustainable energy development.
Public awareness and concern about environmental issues drive demand for
cleaner energy alternatives.
"""

# Mixed content (combination of human and AI characteristics)
MIXED_TEXT = """
I recently read an article about quantum computing, and honestly, my brain
felt like it was trying to process quantum superposition itself. The basic
concept is fascinating though: while classical computers use bits that are
either 0 or 1, quantum computers use qubits that can exist in multiple states
simultaneously. This property, called superposition, allows quantum computers
to perform certain calculations much faster than classical computers.

Quantum computing has potential applications in cryptography, drug discovery,
and optimization problems. Companies like IBM, Google, and Microsoft are
investing heavily in quantum research. The technology is still in early stages,
but progress has been remarkable.

Still, I can't help but wonder - will my laptop need a quantum processor someday
just to run email? Technology has a way of making yesterday's supercomputers
look like pocket calculators. My first computer had 256MB of RAM and I thought
I was living in the future!
"""

# Very short snippets for edge case testing
SHORT_SNIPPETS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world!",
    "Machine learning is a subset of artificial intelligence.",
    "I love pizza.",
    "The weather today is quite pleasant.",
]
