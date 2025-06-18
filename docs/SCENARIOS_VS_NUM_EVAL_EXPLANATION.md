# Scenarios vs num_eval: Clear Distinction and Importance

## 🤔 The Question

> "Is the scenarios anyhow meaningful since we can simply extend num_eval? What advantage do scenarios give?"

## ✅ **Answer: Yes, scenarios are very meaningful and serve a completely different purpose than num_eval.**

## 📊 Clear Distinction

### **Scenarios** = Different Time Periods/Situations
- **Purpose**: Test algorithm performance across **different market conditions**
- **Example**: 120 scenarios = every 5 minutes from 10:00-20:00 (different times of day)
- **What varies**: Request/driver numbers, time-of-day effects, market dynamics
- **From original paper**: "120 situations are simulated" (different time periods)

### **num_eval** = Monte Carlo Evaluations per Scenario  
- **Purpose**: Reduce **randomness/noise** within each scenario
- **Example**: 100 evaluations = run same scenario 100 times with different random seeds
- **What varies**: Random matching outcomes, acceptance probability realizations
- **From original paper**: "N := 102" (Monte Carlo evaluations)

## 🧪 Practical Example

```python
# Original experiment_PL.py structure
for tt_tmp in range(simulation_range):  # scenarios = different time periods
    # ... generate scenario data for time period tt_tmp ...
    
    for k in range(num_eval):           # num_eval = Monte Carlo per scenario
        # ... random acceptance decisions ...
        # ... random matching outcomes ...
        objective_value_proposed_list[k] = result
    
    # Average across num_eval for this scenario
    scenario_result = np.mean(objective_value_proposed_list)
```

### **Scenario 1**: 10:00 AM (Rush hour)
- High request/driver numbers: 120 requests, 90 drivers
- **num_eval=100**: Run this scenario 100 times with different random outcomes
- **Result**: Average objective value = 1250.45 ± 45.23

### **Scenario 2**: 2:00 PM (Low demand)  
- Low request/driver numbers: 40 requests, 35 drivers
- **num_eval=100**: Run this scenario 100 times with different random outcomes
- **Result**: Average objective value = 485.67 ± 15.89

### **Scenario 3**: 6:00 PM (Evening rush)
- Very high request/driver numbers: 180 requests, 110 drivers  
- **num_eval=100**: Run this scenario 100 times with different random outcomes
- **Result**: Average objective value = 1856.32 ± 67.14

## 🎯 Why Both Are Essential

### **Without Scenarios** (only extending num_eval):
```bash
# This would be WRONG:
./run_experiment.sh ... simulation_range=1 num_eval=12000
```
- ❌ **Only tests ONE market condition** (e.g., only 10:00 AM)
- ❌ **Doesn't capture time-of-day variations**
- ❌ **Doesn't test algorithm robustness across different situations**
- ❌ **Not comparable to original paper results**

### **With Proper Scenarios + num_eval**:
```bash
# This is CORRECT:
./run_experiment.sh ... simulation_range=120 num_eval=100
```
- ✅ **Tests 120 different market conditions** (every 5 minutes, 10:00-20:00)
- ✅ **Each condition tested robustly** (100 Monte Carlo runs)
- ✅ **Captures realistic performance variation**
- ✅ **Comparable to original paper methodology**

## 📈 Performance Analysis Benefits

### **Scenarios Enable**:
1. **Time-of-day analysis**: "Algorithm X performs 15% better during rush hours"
2. **Market condition robustness**: "Algorithm Y struggles when demand >> supply"
3. **Realistic performance ranges**: "Expected daily revenue: $1200-1800"
4. **Comparative insights**: "Linear Program outperforms MAPS consistently across all time periods"

### **num_eval Enables**:
1. **Statistical confidence**: "Result is significant with 95% confidence"
2. **Noise reduction**: "True performance ± measurement uncertainty"
3. **Reliable comparisons**: "Method A is actually better than Method B (not just random variation)"

## 🔬 Original Paper Compliance

From the Hikima paper:
> "We perform simulations using the data from October 6 and 10, 2019... matching situations are constructed **every 5 minutes** from 10:00 to 20:00, that is, **120 situations are simulated**."

> "We define ER := 1/N ∑(k=1 to N) max f(x,z), where N := 10²"

**Translation**:
- **120 situations** = 120 scenarios (different time periods)
- **N := 10²** = 100 Monte Carlo evaluations per scenario

## 🚀 Our Implementation

```python
def _run_single_day(self, method, month, day):
    for scenario in range(self.simulation_range):  # 120 scenarios
        # Generate scenario data for this time period
        hour, minute = calculate_time_from_scenario(scenario)
        num_requests, num_drivers = generate_scenario_data(hour, minute)
        
        # Run num_eval Monte Carlo evaluations for this scenario
        scenario_objectives = []
        for eval_iteration in range(self.num_eval):  # 100 evaluations
            result = run_method_with_random_seed(method, num_requests, num_drivers)
            scenario_objectives.append(result['objective_value'])
        
        # Store average for this scenario
        objectives.append(np.mean(scenario_objectives))
```

## 💡 Command Examples

### **Single Day Experiment**:
```bash
./run_experiment.sh 10 20 5m Manhattan 30s 10 6 2019
# 120 scenarios (every 5 min, 10:00-20:00) × 100 evaluations = 12,000 total runs
# Result: Performance across entire day with statistical confidence
```

### **Multi-Day Experiment**:
```bash  
./run_multi_month.sh 10 20 5m Manhattan 30s "3,4,5" "6,10" 2019
# 120 scenarios × 2 days × 3 months × 100 evaluations = 72,000 total runs
# Result: Seasonal and weekly performance patterns
```

## 📊 Result Structure

```json
{
  "daily_summaries": [
    {
      "date": "2019-10-06",
      "methods": {
        "hikima": {
          "avg_objective_value": 1250.45,    // Average across 120 scenarios
          "std_objective_value": 125.30,     // Variation across time periods  
          "total_scenarios": 120              // Different time periods tested
        }
      }
    }
  ]
}
```

## 🎯 **Conclusion**

**Scenarios and num_eval serve completely different purposes:**
- **Scenarios** = Test different situations (essential for realistic evaluation)
- **num_eval** = Reduce noise within each situation (essential for statistical confidence)

**You cannot substitute one for the other.** Both are needed for meaningful, robust, and comparable results. 