# **Local AI Investment Research Platform**

## **Overview**

The Local AI Investment Research Platform is a fully local, AI-powered stock research and paper-trading system running on an NVIDIA DGX Spark.

The platform analyzes stocks, generates investment research, ranks opportunities, enforces risk controls, creates reports, and simulates trades.

**No real-money trading is currently performed.**

All trades are executed through a simulated paper-trading environment.

---

# **Core Design Philosophy**

The platform follows a strict separation of responsibilities:

Agents analyze.  
Risk engine controls.  
Human approves.  
Broker executes.

AI models generate research and recommendations.

Deterministic risk controls enforce portfolio safety.

Human approval remains the final authority.

---

# **System Architecture**

Market Data  
     |  
     v

Technical Agent  
     |  
     v

Kronos Forecast Agent  
     |  
     v

Fundamentals Agent  
     |  
     v

News Agent  
     |  
     v

SEC Filing Agent  
     |  
     v

Portfolio Agent  
     |  
     v

Trade Decision Agent  
     |  
     v

Risk Engine  
     |  
     v

Paper Trading Engine  
     |  
     v

SQLite Database  
     |  
     v

Reports / Dashboard

---

# **Component Breakdown**

## **1\. Kronos Forecast Agent**

### **Purpose**

Uses Kronos time-series forecasting to predict future stock price behavior.

### **Responsibilities**

* Download historical OHLCV data  
* Generate Kronos-compatible CSV files  
* Load Kronos model  
* Execute forecasts  
* Calculate expected return  
* Generate forecast score  
* Estimate confidence

### **Output**

{  
  "forecast\_score": 74,  
  "forecast\_direction": "bullish",  
  "expected\_return\_pct": 4.2,  
  "forecast\_confidence": 0.62  
}

---

## **2\. Technical Analysis Agent**

### **Purpose**

Performs deterministic technical analysis using Python and pandas.

### **Metrics**

* 50-Day Moving Average  
* 200-Day Moving Average  
* Volatility  
* Momentum  
* Drawdown  
* Trend Strength

### **Output**

{  
  "technical\_score": 90,  
  "trend": "uptrend",  
  "volatility": "moderate"  
}

---

## **3\. Fundamentals Agent**

### **Purpose**

Evaluates overall business quality.

### **Factors**

* Revenue Growth  
* Profitability  
* Balance Sheet Strength  
* Cash Flow  
* Valuation

### **Output**

{  
  "fundamental\_score": 78.25,  
  "growth\_score": 85,  
  "profitability\_score": 100,  
  "balance\_sheet\_score": 55,  
  "cash\_flow\_score": 90,  
  "valuation\_score": 25  
}

---

## **4\. News Agent**

### **Purpose**

Analyzes recent news and market sentiment.

### **Detects**

* Earnings surprises  
* Product launches  
* Regulatory actions  
* Lawsuits  
* Executive changes  
* Analyst actions  
* Major company events

### **Output**

{  
  "news\_score": 82,  
  "sentiment": "positive"  
}

---

## **5\. SEC Filing Agent**

### **Purpose**

Analyzes SEC filings for long-term risk indicators.

### **Supported Documents**

* 10-K  
* 10-Q  
* Future SEC filing types

### **Detects**

* Risk factors  
* Debt concerns  
* Litigation  
* Regulatory exposure  
* Material weaknesses

### **Output**

{  
  "filing\_score": 76,  
  "risk\_terms": 12,  
  "filing\_date": "2026-01-30"  
}

---

## **6\. Portfolio Agent**

### **Purpose**

Determines whether a trade fits within portfolio constraints.

### **Evaluates**

* Position sizing  
* Sector exposure  
* Cash reserves  
* Diversification  
* Allocation limits

### **Output**

{  
  "portfolio\_fit\_score": 90,  
  "approved\_by\_portfolio\_agent": true  
}

---

## **7\. Trade Decision Agent**

### **Purpose**

Combines all research signals into a final recommendation.

### **Current Weighting**

Fundamentals      35%  
Technical         25%  
Kronos Forecast   20%  
News              10%  
SEC Filing        10%

### **Output**

{  
  "final\_score": 65.4,  
  "recommendation": "watchlist",  
  "suggested\_action": "hold"  
}

---

## **8\. Risk Engine**

### **Purpose**

Acts as the platform's safety layer.

### **Characteristics**

* Pure Python  
* Deterministic  
* No AI  
* No LLM decision-making

### **Enforces**

* Position limits  
* Sector limits  
* Cash reserve minimums  
* Minimum score thresholds  
* Asset restrictions  
* Human approval requirements

### **Example Result**

{  
  "approved": false,  
  "errors": \[  
    "Final score below threshold",  
    "Kronos score below threshold"  
  \]  
}

---

## **9\. Report Agent**

### **Purpose**

Generates research reports and summaries.

### **Produces**

* Daily Reports  
* Markdown Reports  
* JSON Summaries  
* SQLite Records  
* Leaderboards

### **Output Location**

reports/daily/

---

# **Data Layer**

## **SQLite Database**

Location:

db/investment\_research.sqlite

### **Tracks**

* Agent scores  
* Trade decisions  
* Paper account  
* Positions  
* Paper trades  
* Historical performance

### **Purpose**

Provides a complete audit trail of every decision.

---

# **Multi-Ticker Scanner**

## **Purpose**

Runs the entire research pipeline against a watchlist.

### **Current Watchlist**

AAPL  
MSFT  
NVDA  
AMZN  
META  
GOOGL  
AVGO  
COST  
V  
BRK-B

### **Responsibilities**

* Execute all agents  
* Rank stocks  
* Generate reports  
* Update database  
* Submit approved paper trades

### **Entry Point**

python scripts/run\_daily\_analysis.py

---

# **Paper Trading Engine**

## **Purpose**

Simulates trading using virtual capital.

### **Current Account**

Starting Cash: $100,000

### **Tracks**

* Cash  
* Positions  
* Trade History  
* Account Equity

### **Requirements**

Trades execute only when:

Trade Decision Approved  
Risk Engine Approved  
Portfolio Rules Approved

---

# **Historical Replay Backtester**

## **Purpose**

Tests strategies against historical data.

### **Features**

* Historical Replay  
* Benchmark Comparison  
* Signal Evaluation  
* Trade Simulation

### **Outputs**

reports/backtests/  
data/backtests/

### **Benchmarks**

SPY  
QQQ

---

# **Scheduler**

## **Purpose**

Automates daily execution of the platform.

### **Current Implementation**

cron

### **Wrapper Script**

run\_daily\_analysis.sh

### **Execution Command**

python scripts/run\_daily\_analysis.py

### **Schedule**

Weekdays after market close

---

# **Current Platform Status**

## **Completed**

✓ Market Data Feed  
✓ Kronos Forecast Agent  
✓ Technical Agent  
✓ Fundamentals Agent  
✓ News Agent  
✓ SEC Filing Agent  
✓ Portfolio Agent  
✓ Trade Decision Agent  
✓ Risk Engine  
✓ Report Agent  
✓ SQLite Logging  
✓ Multi-Ticker Scanner  
✓ Daily Leaderboard  
✓ Historical Backtester  
✓ Paper Trading Engine  
✓ Scheduler Deployment

---

# **Current Data Flow**

Watchlist  
    ↓  
Multi-Ticker Scanner  
    ↓  
Fundamentals Agent  
    ↓  
Technical Agent  
    ↓  
Kronos Forecast Agent  
    ↓  
News Agent  
    ↓  
SEC Filing Agent  
    ↓  
Portfolio Agent  
    ↓  
Trade Decision Agent  
    ↓  
Risk Engine  
    ↓  
Paper Trader  
    ↓  
SQLite Database  
    ↓  
Daily Reports

---

# **Next Development Priorities**

## **1\. Portfolio Performance Tracker**

Track:

* Account Equity  
* Cash Balance  
* Open Positions  
* Unrealized Gain/Loss  
* Realized Gain/Loss

---

## **2\. Performance Analytics**

Measure:

* Win Rate  
* Average Return  
* Sharpe Ratio  
* Maximum Drawdown  
* Benchmark Comparison

---

## **3\. Dashboard**

Display:

* Leaderboard  
* Positions  
* Performance Metrics  
* Reports  
* Risk Status  
* Historical Results

---

## **4\. Alpaca Paper Broker Integration**

Add:

* Paper Orders  
* Position Sync  
* Order Tracking

No live trading should be enabled until extensive validation has been completed.

---

# **Current Project Stage**

Signal Generation      → Complete  
Research Platform      → Complete  
Paper Trading          → Deployed  
Performance Tracking   → Next  
Analytics Dashboard    → Next  
Live Trading           → Future

# **Summary**

The Local AI Investment Research Platform is a multi-agent stock research system that combines market forecasting, technical analysis, fundamentals, news analysis, SEC filing review, portfolio management, and deterministic risk controls into a single automated workflow.

The platform currently generates investment research, ranks opportunities, enforces risk policies, logs results, and simulates trades while remaining fully local and under human control.

