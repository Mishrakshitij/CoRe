"""
Generate SVG figures for the CoRe paper.
These can be imported into draw.io, Mural, or used directly.

Usage:
    python generate_figures.py

Output:
    figures/framework_overview.svg
    figures/rescue_mechanism.svg
    figures/training_flow.svg
"""

import os

# Create figures directory
os.makedirs("figures", exist_ok=True)


def create_framework_overview():
    """Main framework overview figure showing Round A and Round B."""

    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500" width="800" height="500">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
    <marker id="arrowhead-green" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#27ae60"/>
    </marker>
    <marker id="arrowhead-red" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#e74c3c"/>
    </marker>
    <linearGradient id="grad-blue" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#3498db;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#2980b9;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad-purple" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#9b59b6;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#8e44ad;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad-green" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#27ae60;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#1e8449;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad-orange" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#f39c12;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#d68910;stop-opacity:1" />
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="800" height="500" fill="#fafafa"/>

  <!-- Title -->
  <text x="400" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#2c3e50">
    Collaborative Reasoning (CoRe) Framework
  </text>

  <!-- Problem Input -->
  <rect x="30" y="200" width="100" height="60" rx="8" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="2"/>
  <text x="80" y="225" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#2c3e50">Problem</text>
  <text x="80" y="245" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#7f8c8d">x</text>

  <!-- Arrow to models -->
  <line x1="130" y1="230" x2="170" y2="150" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="130" y1="230" x2="170" y2="310" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>

  <!-- Round A Section -->
  <rect x="160" y="60" width="280" height="180" rx="10" fill="none" stroke="#3498db" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="300" y="85" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#3498db">Round A: Cold Generation</text>

  <!-- Model M1 -->
  <rect x="180" y="100" width="100" height="50" rx="8" fill="url(#grad-blue)" stroke="#2980b9" stroke-width="2"/>
  <text x="230" y="130" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="white">M₁</text>

  <!-- Model M2 -->
  <rect x="180" y="170" width="100" height="50" rx="8" fill="url(#grad-purple)" stroke="#8e44ad" stroke-width="2"/>
  <text x="230" y="200" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="white">M₂</text>

  <!-- K traces arrows -->
  <line x1="280" y1="125" x2="320" y2="125" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="280" y1="195" x2="320" y2="195" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  <text x="300" y="115" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#7f8c8d">K traces</text>
  <text x="300" y="185" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#7f8c8d">K traces</text>

  <!-- M1 Output (Failed) -->
  <rect x="330" y="100" width="90" height="50" rx="8" fill="#fadbd8" stroke="#e74c3c" stroke-width="2"/>
  <text x="375" y="122" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#c0392b">y₁: ✗</text>
  <text x="375" y="138" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#e74c3c">Failed</text>

  <!-- M2 Output (Success) -->
  <rect x="330" y="170" width="90" height="50" rx="8" fill="#d5f5e3" stroke="#27ae60" stroke-width="2"/>
  <text x="375" y="192" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#1e8449">y₂: ✓</text>
  <text x="375" y="208" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#27ae60">Success</text>

  <!-- Round B Section -->
  <rect x="160" y="260" width="280" height="180" rx="10" fill="none" stroke="#f39c12" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="300" y="285" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#f39c12">Round B: Contexted Generation</text>

  <!-- Hint Construction -->
  <rect x="180" y="300" width="100" height="50" rx="8" fill="url(#grad-orange)" stroke="#d68910" stroke-width="2"/>
  <text x="230" y="320" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="white">Hint</text>
  <text x="230" y="338" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="white">h₂→₁</text>

  <!-- Arrow from M2 success to Hint -->
  <path d="M 375 220 Q 375 260, 280 300" fill="none" stroke="#27ae60" stroke-width="2" marker-end="url(#arrowhead-green)"/>
  <text x="340" y="265" font-family="Arial, sans-serif" font-size="9" fill="#27ae60">Extract hint</text>

  <!-- M1 Rescue -->
  <rect x="180" y="370" width="100" height="50" rx="8" fill="url(#grad-blue)" stroke="#2980b9" stroke-width="2"/>
  <text x="230" y="392" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="white">M₁</text>
  <text x="230" y="408" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="white">+ hint</text>

  <!-- Arrow hint to M1 -->
  <line x1="230" y1="350" x2="230" y2="365" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>

  <!-- K' traces arrow -->
  <line x1="280" y1="395" x2="320" y2="395" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  <text x="300" y="385" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#7f8c8d">K' traces</text>

  <!-- Rescue Output -->
  <rect x="330" y="370" width="90" height="50" rx="8" fill="url(#grad-green)" stroke="#1e8449" stroke-width="2"/>
  <text x="375" y="392" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="white">ỹ₁: ✓</text>
  <text x="375" y="408" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="white">Rescued!</text>

  <!-- Policy Optimization Section -->
  <rect x="480" y="100" width="280" height="350" rx="10" fill="none" stroke="#9b59b6" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="620" y="125" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#9b59b6">Policy Optimization</text>

  <!-- Arrows to optimization -->
  <line x1="420" y1="125" x2="490" y2="180" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="420" y1="195" x2="490" y2="210" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="420" y1="395" x2="490" y2="350" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>

  <!-- Reward Components -->
  <rect x="500" y="150" width="120" height="40" rx="5" fill="#e8f6f3" stroke="#1abc9c" stroke-width="1"/>
  <text x="560" y="175" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#16a085">R_exploit</text>

  <rect x="630" y="150" width="120" height="40" rx="5" fill="#fef9e7" stroke="#f1c40f" stroke-width="1"/>
  <text x="690" y="175" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#d4ac0d">R_explore</text>

  <rect x="500" y="200" width="120" height="40" rx="5" fill="#f5eef8" stroke="#9b59b6" stroke-width="1"/>
  <text x="560" y="225" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#8e44ad">R_cross</text>

  <!-- Combined Reward -->
  <rect x="540" y="260" width="160" height="45" rx="8" fill="#2c3e50" stroke="#1a252f" stroke-width="2"/>
  <text x="620" y="280" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="white">Combined Reward</text>
  <text x="620" y="295" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#bdc3c7">R = w₁R_e + w₂R_x + w₃R_c</text>

  <!-- Arrows to combined -->
  <line x1="560" y1="190" x2="600" y2="255" stroke="#95a5a6" stroke-width="1"/>
  <line x1="690" y1="190" x2="650" y2="255" stroke="#95a5a6" stroke-width="1"/>
  <line x1="560" y1="240" x2="590" y2="255" stroke="#95a5a6" stroke-width="1"/>

  <!-- Advantage Computation -->
  <rect x="540" y="320" width="160" height="40" rx="8" fill="#ebedef" stroke="#bdc3c7" stroke-width="2"/>
  <text x="620" y="345" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#2c3e50">Â = (R - μ) / σ</text>

  <line x1="620" y1="305" x2="620" y2="315" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>

  <!-- Algorithm Options -->
  <rect x="500" y="380" width="70" height="35" rx="5" fill="#3498db" stroke="#2980b9" stroke-width="1"/>
  <text x="535" y="402" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="white">GRPO</text>

  <rect x="580" y="380" width="70" height="35" rx="5" fill="#e74c3c" stroke="#c0392b" stroke-width="1"/>
  <text x="615" y="402" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="white">GSPO</text>

  <rect x="660" y="380" width="70" height="35" rx="5" fill="#27ae60" stroke="#1e8449" stroke-width="1"/>
  <text x="695" y="402" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="white">SAPO</text>

  <line x1="620" y1="360" x2="620" y2="375" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>

  <!-- Update arrows back to models -->
  <path d="M 535 415 Q 500 450, 250 430 Q 150 420, 180 200" fill="none" stroke="#3498db" stroke-width="2" stroke-dasharray="4,2" marker-end="url(#arrowhead)"/>
  <text x="100" y="380" font-family="Arial, sans-serif" font-size="10" fill="#3498db">Update θ</text>

  <!-- Legend -->
  <rect x="580" y="440" width="180" height="50" rx="5" fill="white" stroke="#bdc3c7" stroke-width="1"/>
  <text x="670" y="455" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" font-weight="bold" fill="#7f8c8d">Legend</text>
  <circle cx="600" cy="470" r="5" fill="#27ae60"/>
  <text x="615" y="473" font-family="Arial, sans-serif" font-size="8" fill="#2c3e50">Success</text>
  <circle cx="680" cy="470" r="5" fill="#e74c3c"/>
  <text x="695" y="473" font-family="Arial, sans-serif" font-size="8" fill="#2c3e50">Fail</text>
  <circle cx="740" cy="470" r="5" fill="#f39c12"/>
  <text x="755" y="473" font-family="Arial, sans-serif" font-size="8" fill="#2c3e50">Rescue</text>
</svg>'''

    with open("figures/framework_overview.svg", "w") as f:
        f.write(svg)
    print("Created: figures/framework_overview.svg")


def create_algorithm_comparison():
    """Figure comparing GRPO, GSPO, and SAPO clipping/gating."""

    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 300" width="800" height="300">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <!-- Background -->
  <rect width="800" height="300" fill="#fafafa"/>

  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#2c3e50">
    Policy Optimization Algorithm Comparison
  </text>

  <!-- GRPO Section -->
  <rect x="20" y="50" width="240" height="230" rx="10" fill="white" stroke="#3498db" stroke-width="2"/>
  <text x="140" y="75" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#3498db">GRPO</text>
  <text x="140" y="92" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#7f8c8d">Token-level Hard Clipping</text>

  <!-- GRPO Graph -->
  <line x1="50" y1="220" x2="220" y2="220" stroke="#333" stroke-width="1" marker-end="url(#arrow)"/>
  <line x1="135" y1="250" x2="135" y2="110" stroke="#333" stroke-width="1" marker-end="url(#arrow)"/>
  <text x="225" y="225" font-family="Arial, sans-serif" font-size="9" fill="#333">r_t</text>
  <text x="140" y="105" font-family="Arial, sans-serif" font-size="9" fill="#333">weight</text>

  <!-- GRPO clipping function -->
  <path d="M 50 180 L 100 180 L 100 140 L 170 140 L 170 180 L 220 180" fill="none" stroke="#3498db" stroke-width="3"/>
  <text x="100" y="240" font-family="Arial, sans-serif" font-size="8" fill="#7f8c8d">1-ε</text>
  <text x="165" y="240" font-family="Arial, sans-serif" font-size="8" fill="#7f8c8d">1+ε</text>
  <line x1="100" y1="218" x2="100" y2="222" stroke="#333" stroke-width="1"/>
  <line x1="170" y1="218" x2="170" y2="222" stroke="#333" stroke-width="1"/>

  <text x="140" y="265" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#2c3e50">ε = 0.2</text>

  <!-- GSPO Section -->
  <rect x="280" y="50" width="240" height="230" rx="10" fill="white" stroke="#e74c3c" stroke-width="2"/>
  <text x="400" y="75" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#e74c3c">GSPO</text>
  <text x="400" y="92" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#7f8c8d">Sequence-level Tight Clipping</text>

  <!-- GSPO Graph -->
  <line x1="310" y1="220" x2="480" y2="220" stroke="#333" stroke-width="1" marker-end="url(#arrow)"/>
  <line x1="395" y1="250" x2="395" y2="110" stroke="#333" stroke-width="1" marker-end="url(#arrow)"/>
  <text x="485" y="225" font-family="Arial, sans-serif" font-size="9" fill="#333">w_seq</text>
  <text x="400" y="105" font-family="Arial, sans-serif" font-size="9" fill="#333">weight</text>

  <!-- GSPO tight clipping function -->
  <path d="M 310 160 L 390 160 L 390 140 L 400 140 L 400 160 L 480 160" fill="none" stroke="#e74c3c" stroke-width="3"/>
  <text x="388" y="240" font-family="Arial, sans-serif" font-size="7" fill="#7f8c8d">1-ε</text>
  <text x="398" y="240" font-family="Arial, sans-serif" font-size="7" fill="#7f8c8d">1+ε</text>
  <line x1="390" y1="218" x2="390" y2="222" stroke="#333" stroke-width="1"/>
  <line x1="400" y1="218" x2="400" y2="222" stroke="#333" stroke-width="1"/>

  <text x="400" y="265" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#2c3e50">ε ≈ 3×10⁻⁴</text>

  <!-- SAPO Section -->
  <rect x="540" y="50" width="240" height="230" rx="10" fill="white" stroke="#27ae60" stroke-width="2"/>
  <text x="660" y="75" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#27ae60">SAPO</text>
  <text x="660" y="92" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#7f8c8d">Soft Sigmoid Gating</text>

  <!-- SAPO Graph -->
  <line x1="570" y1="220" x2="740" y2="220" stroke="#333" stroke-width="1" marker-end="url(#arrow)"/>
  <line x1="655" y1="250" x2="655" y2="110" stroke="#333" stroke-width="1" marker-end="url(#arrow)"/>
  <text x="745" y="225" font-family="Arial, sans-serif" font-size="9" fill="#333">r_t</text>
  <text x="660" y="105" font-family="Arial, sans-serif" font-size="9" fill="#333">g_t</text>

  <!-- SAPO sigmoid function (smooth curve) -->
  <path d="M 570 200 Q 600 200, 620 180 Q 640 155, 655 145 Q 670 155, 690 180 Q 710 200, 740 200" fill="none" stroke="#27ae60" stroke-width="3"/>
  <text x="655" y="240" font-family="Arial, sans-serif" font-size="8" fill="#7f8c8d">1</text>
  <line x1="655" y1="218" x2="655" y2="222" stroke="#333" stroke-width="1"/>

  <text x="660" y="265" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#2c3e50">g = σ(τ(r-1)) · 4/τ</text>
</svg>'''

    with open("figures/algorithm_comparison.svg", "w") as f:
        f.write(svg)
    print("Created: figures/algorithm_comparison.svg")


def create_rescue_flow():
    """Detailed rescue mechanism flow."""

    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 350" width="700" height="350">
  <defs>
    <marker id="arr" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
    <marker id="arr-green" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#27ae60"/>
    </marker>
  </defs>

  <!-- Background -->
  <rect width="700" height="350" fill="#fafafa"/>

  <!-- Title -->
  <text x="350" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#2c3e50">
    Cross-Model Rescue Mechanism
  </text>

  <!-- Step 1: Problem -->
  <rect x="30" y="80" width="100" height="60" rx="8" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="2"/>
  <text x="80" y="105" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#2c3e50">Problem x</text>
  <text x="80" y="125" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#7f8c8d">"What is 15% of 80?"</text>

  <!-- Step 2: M1 generates (fails) -->
  <rect x="170" y="60" width="120" height="45" rx="8" fill="#fadbd8" stroke="#e74c3c" stroke-width="2"/>
  <text x="230" y="80" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="#c0392b">M₁ generates</text>
  <text x="230" y="95" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#e74c3c">y₁ = "15" ✗</text>

  <!-- Step 2: M2 generates (succeeds) -->
  <rect x="170" y="115" width="120" height="45" rx="8" fill="#d5f5e3" stroke="#27ae60" stroke-width="2"/>
  <text x="230" y="135" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="#1e8449">M₂ generates</text>
  <text x="230" y="150" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#27ae60">y₂ = "12" ✓</text>

  <!-- Arrows from problem -->
  <line x1="130" y1="100" x2="165" y2="82" stroke="#333" stroke-width="2" marker-end="url(#arr)"/>
  <line x1="130" y1="120" x2="165" y2="137" stroke="#333" stroke-width="2" marker-end="url(#arr)"/>

  <!-- Step 3: Hint extraction -->
  <rect x="330" y="115" width="120" height="60" rx="8" fill="#fef9e7" stroke="#f39c12" stroke-width="2"/>
  <text x="390" y="135" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="#d68910">Hint Extraction</text>
  <text x="390" y="152" text-anchor="middle" font-family="Arial, sans-serif" font-size="8" fill="#7f8c8d">Compress(y₂)</text>
  <text x="390" y="167" text-anchor="middle" font-family="Arial, sans-serif" font-size="8" fill="#b7950b">"Multiply 80×0.15"</text>

  <!-- Arrow from M2 to hint -->
  <line x1="290" y1="137" x2="325" y2="140" stroke="#27ae60" stroke-width="2" marker-end="url(#arr-green)"/>

  <!-- Step 4: Rescue prompt -->
  <rect x="330" y="200" width="120" height="70" rx="8" fill="#e8daef" stroke="#9b59b6" stroke-width="2"/>
  <text x="390" y="220" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="#8e44ad">Rescue Prompt</text>
  <text x="390" y="238" text-anchor="middle" font-family="Arial, sans-serif" font-size="8" fill="#7f8c8d">x + &lt;hint&gt;</text>
  <text x="390" y="253" text-anchor="middle" font-family="Arial, sans-serif" font-size="7" fill="#9b59b6">"What is 15% of 80?</text>
  <text x="390" y="263" text-anchor="middle" font-family="Arial, sans-serif" font-size="7" fill="#9b59b6">Hint: Multiply 80×0.15"</text>

  <!-- Arrow from hint to rescue prompt -->
  <line x1="390" y1="175" x2="390" y2="195" stroke="#333" stroke-width="2" marker-end="url(#arr)"/>

  <!-- Step 5: M1 rescue generation -->
  <rect x="490" y="200" width="120" height="70" rx="8" fill="#d6eaf8" stroke="#3498db" stroke-width="2"/>
  <text x="550" y="220" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="#2980b9">M₁ Rescue</text>
  <text x="550" y="240" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#5dade2">K' traces with hint</text>
  <text x="550" y="260" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#3498db">ỹ₁ = "12" ✓</text>

  <!-- Arrow from rescue prompt to M1 rescue -->
  <line x1="450" y1="235" x2="485" y2="235" stroke="#333" stroke-width="2" marker-end="url(#arr)"/>

  <!-- Step 6: Result -->
  <rect x="490" y="290" width="120" height="45" rx="8" fill="#27ae60" stroke="#1e8449" stroke-width="2"/>
  <text x="550" y="310" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="white">Rescued!</text>
  <text x="550" y="325" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="white">R_cross bonus</text>

  <!-- Arrow to result -->
  <line x1="550" y1="270" x2="550" y2="285" stroke="#333" stroke-width="2" marker-end="url(#arr)"/>

  <!-- Condition box -->
  <rect x="170" y="190" width="120" height="40" rx="5" fill="#f8f9fa" stroke="#dee2e6" stroke-width="1"/>
  <text x="230" y="205" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#6c757d">Condition:</text>
  <text x="230" y="220" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#495057">s₁=0 ∧ s₂=1</text>

  <!-- Arrow from condition to rescue prompt -->
  <line x1="290" y1="210" x2="325" y2="220" stroke="#7f8c8d" stroke-width="1" stroke-dasharray="4,2"/>

  <!-- Step numbers -->
  <circle cx="80" cy="55" r="12" fill="#3498db"/>
  <text x="80" y="59" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="white">1</text>

  <circle cx="230" cy="40" r="12" fill="#3498db"/>
  <text x="230" y="44" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="white">2</text>

  <circle cx="390" cy="95" r="12" fill="#3498db"/>
  <text x="390" y="99" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="white">3</text>

  <circle cx="550" cy="180" r="12" fill="#3498db"/>
  <text x="550" y="184" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="white">4</text>
</svg>'''

    with open("figures/rescue_mechanism.svg", "w") as f:
        f.write(svg)
    print("Created: figures/rescue_mechanism.svg")


def create_training_curves_placeholder():
    """Placeholder for training curves - would normally use matplotlib."""

    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400" width="600" height="400">
  <!-- Background -->
  <rect width="600" height="400" fill="white"/>

  <!-- Title -->
  <text x="300" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#2c3e50">
    Training Dynamics
  </text>

  <!-- Accuracy subplot -->
  <rect x="50" y="50" width="230" height="150" fill="#fafafa" stroke="#ddd" stroke-width="1"/>
  <text x="165" y="70" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#333">Accuracy (%)</text>

  <!-- Axis -->
  <line x1="70" y1="180" x2="260" y2="180" stroke="#333" stroke-width="1"/>
  <line x1="70" y1="180" x2="70" y2="70" stroke="#333" stroke-width="1"/>

  <!-- Placeholder curves -->
  <path d="M 70 170 Q 100 160, 130 145 Q 160 130, 190 115 Q 220 105, 250 95" fill="none" stroke="#3498db" stroke-width="2"/>
  <path d="M 70 175 Q 100 170, 130 160 Q 160 150, 190 140 Q 220 135, 250 130" fill="none" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,3"/>

  <!-- Legend -->
  <line x1="90" y1="195" x2="110" y2="195" stroke="#3498db" stroke-width="2"/>
  <text x="115" y="198" font-family="Arial, sans-serif" font-size="9" fill="#333">CoRe</text>
  <line x1="160" y1="195" x2="180" y2="195" stroke="#e74c3c" stroke-width="2" stroke-dasharray="5,3"/>
  <text x="185" y="198" font-family="Arial, sans-serif" font-size="9" fill="#333">Independent</text>

  <!-- Loss subplot -->
  <rect x="320" y="50" width="230" height="150" fill="#fafafa" stroke="#ddd" stroke-width="1"/>
  <text x="435" y="70" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#333">Loss</text>

  <!-- Axis -->
  <line x1="340" y1="180" x2="530" y2="180" stroke="#333" stroke-width="1"/>
  <line x1="340" y1="180" x2="340" y2="70" stroke="#333" stroke-width="1"/>

  <!-- Placeholder curves -->
  <path d="M 340 90 Q 380 110, 420 130 Q 460 145, 500 155 Q 520 160, 530 165" fill="none" stroke="#27ae60" stroke-width="2"/>

  <!-- Rescue Rate subplot -->
  <rect x="50" y="230" width="230" height="150" fill="#fafafa" stroke="#ddd" stroke-width="1"/>
  <text x="165" y="250" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#333">Rescue Rate (%)</text>

  <!-- Axis -->
  <line x1="70" y1="360" x2="260" y2="360" stroke="#333" stroke-width="1"/>
  <line x1="70" y1="360" x2="70" y2="250" stroke="#333" stroke-width="1"/>

  <!-- Placeholder curve -->
  <path d="M 70 350 Q 100 330, 130 310 Q 160 295, 190 285 Q 220 280, 250 280" fill="none" stroke="#9b59b6" stroke-width="2"/>

  <!-- X-axis label -->
  <text x="165" y="385" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Training Steps</text>

  <!-- Note -->
  <rect x="320" y="230" width="230" height="150" fill="#f8f9fa" stroke="#ddd" stroke-width="1"/>
  <text x="435" y="290" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">
    <tspan x="435" dy="0">Replace with actual</tspan>
    <tspan x="435" dy="15">matplotlib/seaborn plots</tspan>
    <tspan x="435" dy="15">from training logs</tspan>
  </text>
</svg>'''

    with open("figures/training_curves.svg", "w") as f:
        f.write(svg)
    print("Created: figures/training_curves.svg")


def create_drawio_xml():
    """Create draw.io compatible XML file."""

    xml = '''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2024-01-01T00:00:00.000Z" agent="Mozilla/5.0" version="22.0.0" type="device">
  <diagram name="CoRe Framework" id="framework">
    <mxGraphModel dx="1434" dy="780" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- Problem Input -->
        <mxCell id="problem" value="Problem&#xa;x" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="40" y="200" width="100" height="60" as="geometry" />
        </mxCell>

        <!-- Model M1 -->
        <mxCell id="m1" value="M₁" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontStyle=1;fontSize=14" vertex="1" parent="1">
          <mxGeometry x="200" y="100" width="100" height="50" as="geometry" />
        </mxCell>

        <!-- Model M2 -->
        <mxCell id="m2" value="M₂" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontStyle=1;fontSize=14" vertex="1" parent="1">
          <mxGeometry x="200" y="280" width="100" height="50" as="geometry" />
        </mxCell>

        <!-- M1 Output Failed -->
        <mxCell id="m1out" value="y₁: ✗&#xa;Failed" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450" vertex="1" parent="1">
          <mxGeometry x="360" y="100" width="80" height="50" as="geometry" />
        </mxCell>

        <!-- M2 Output Success -->
        <mxCell id="m2out" value="y₂: ✓&#xa;Success" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366" vertex="1" parent="1">
          <mxGeometry x="360" y="280" width="80" height="50" as="geometry" />
        </mxCell>

        <!-- Hint -->
        <mxCell id="hint" value="Hint&#xa;h₂→₁" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="360" y="190" width="80" height="50" as="geometry" />
        </mxCell>

        <!-- Rescue -->
        <mxCell id="rescue" value="M₁ + hint&#xa;Rescue" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf" vertex="1" parent="1">
          <mxGeometry x="500" y="190" width="100" height="50" as="geometry" />
        </mxCell>

        <!-- Rescue Output -->
        <mxCell id="rescueout" value="ỹ₁: ✓&#xa;Rescued!" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="660" y="190" width="80" height="50" as="geometry" />
        </mxCell>

        <!-- Arrows -->
        <mxCell id="arr1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="problem" target="m1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="arr2" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="problem" target="m2">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="arr3" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;" edge="1" parent="1" source="m1" target="m1out">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="arr4" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;" edge="1" parent="1" source="m2" target="m2out">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="arr5" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#82b366;" edge="1" parent="1" source="m2out" target="hint">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="arr6" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;" edge="1" parent="1" source="hint" target="rescue">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="arr7" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;" edge="1" parent="1" source="rescue" target="rescueout">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- Labels -->
        <mxCell id="label1" value="Round A: Cold Generation" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontStyle=1;fontColor=#6c8ebf;" vertex="1" parent="1">
          <mxGeometry x="200" y="60" width="160" height="30" as="geometry" />
        </mxCell>
        <mxCell id="label2" value="Round B: Rescue" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontStyle=1;fontColor=#d6b656;" vertex="1" parent="1">
          <mxGeometry x="500" y="150" width="120" height="30" as="geometry" />
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

    with open("figures/framework.drawio", "w") as f:
        f.write(xml)
    print("Created: figures/framework.drawio (importable to draw.io)")


if __name__ == "__main__":
    print("Generating figures for CoRe paper...")
    print("=" * 50)

    create_framework_overview()
    create_algorithm_comparison()
    create_rescue_flow()
    create_training_curves_placeholder()
    create_drawio_xml()

    print("=" * 50)
    print("\nAll figures generated in 'figures/' directory!")
    print("\nTo use in draw.io:")
    print("  1. Open draw.io (app.diagrams.net)")
    print("  2. File > Import From > Device")
    print("  3. Select any .svg or .drawio file")
    print("\nTo use in Mural:")
    print("  1. Upload .svg files directly")
    print("  2. Or copy/paste from draw.io")
