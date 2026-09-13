import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ==========================================
# 1. MATHEMATICAL ANALYSIS & SIMULATION
# ==========================================
p_normal, p_jackpot, p_loss = 0.49, 0.01, 0.50
payout_normal, payout_jackpot = 1000, 10000
ticket_price = 520
flat_fee = 10
tax_rate = 0.25
initial_bankroll = 2000
num_games = 150

# Calculate returns
ret_normal = payout_normal - ticket_price - flat_fee
ret_jackpot = payout_jackpot - ticket_price - flat_fee
ret_loss = -ticket_price - flat_fee

# Skewness Calculation
returns = np.array([ret_normal, ret_jackpot, ret_loss])
probs = np.array([p_normal, p_jackpot, p_loss])
mean = np.sum(returns * probs)
variance = np.sum(probs * (returns - mean)**2)
std_dev = np.sqrt(variance)
skewness = np.sum(probs * ((returns - mean) / std_dev)**3)

# Run Matplotlib Time-Series Simulation
plt.figure(figsize=(10, 5))
for _ in range(50):
    balance = [initial_bankroll]
    for _ in range(num_games):
        if balance[-1] < (ticket_price + flat_fee):
            balance.append(balance[-1])
            continue
        roll = np.random.rand()
        if roll < p_jackpot:
            # Jackpot win (Apply tax to net jackpot profit if positive)
            profit = payout_jackpot - ticket_price - flat_fee
            balance.append(balance[-1] + (profit * (1 - tax_rate) if profit > 0 else profit))
        elif roll < (p_jackpot + p_normal):
            # Normal win
            profit = payout_normal - ticket_price - flat_fee
            balance.append(balance[-1] + (profit * (1 - tax_rate) if profit > 0 else profit))
        else:
            balance.append(balance[-1] + ret_loss)
    plt.plot(balance, alpha=0.3, color='purple' if max(balance) > 5000 else 'gray')

plt.title(f"Jackpot Game Time Series (Skewness: {skewness:.2f})")
plt.xlabel("Number of Games Played")
plt.ylabel("Player Balance ($)")
plt.grid(True, alpha=0.3)
plt.savefig("jackpot_simulation.png", dpi=300)
plt.close()

# ==========================================
# 2. PDF REPORT GENERATION
# ==========================================
pdf_filename = "Skewness_and_Risk_Analysis.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
styles = getSampleStyleSheet()

# Custom Styles
title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor("#1A2B4C"), spaceAfter=15)
body_style = ParagraphStyle('DocBody', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor("#333333"), spaceAfter=10)

story = []
story.append(Paragraph("Risk & Skewness Analysis Report", title_style))
story.append(Spacer(1, 10))

# Document Body Text
intro_text = (
    f"This analysis outlines the mathematical profile of the lottery game featuring a rare jackpot event. "
    f"The profile demonstrates extreme <b>positive skewness ({skewness:.2f})</b>, indicating that the game's "
    f"returns are dominated by infrequent, large-scale payouts rather than standard outcomes."
)
story.append(Paragraph(intro_text, body_style))

# Metrics Table Data
data = [
    ["Metric", "Value"],
    ["Expected Value (EV)", f"${mean:.2f}"],
    ["Standard Deviation", f"${std_dev:.2f}"],
    ["Statistical Skewness", f"{skewness:.2f}"],
    ["Breakeven Ticket Price Cap", "$590.00"]
]

t = Table(data, colWidths=[200, 150])
t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (1,0), colors.HexColor("#1A2B4C")),
    ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('BOTTOMPADDING', (0,0), (-1,0), 8),
    ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F4F6F9")),
    ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
    ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,-1), 10),
]))
story.append(t)
story.append(Spacer(1, 15))

# Impact Summary Text
summary_text = (
    "<b>Key Dynamic Shifts:</b><br/>"
    "1. Flat fees act as a structural hurdle that penalizes survival time directly.<br/>"
    "2. Percentage taxes crush compounding speed by capping peak win distributions.<br/>"
    "3. High skewness extends your victory timeline, making early-game ruin highly probable unless supported by deep capital."
)
story.append(Paragraph(summary_text, body_style))

# Build Document
doc.build(story)
print(f"Success! Saved '{pdf_filename}' and 'jackpot_simulation.png' to your working directory.")
