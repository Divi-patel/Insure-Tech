import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Renewable Energy Insurance Pricing Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

class SolarInsurancePricingApp:
    def __init__(self):
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            st.session_state.data = None
            st.session_state.analysis_data = None
            st.session_state.monthly_stats = {}
            st.session_state.distribution_results = {}
            st.session_state.selected_distributions = {}
            st.session_state.cvar_results = {}
            st.session_state.generation_type = 'Renewable Energy'
            st.session_state.generation_col = 'Generation (MWh)'
            st.session_state.site_registry = None
            st.session_state.facility_capacity = None
            
    def load_data_section(self):
        """Handle data loading section"""
        st.header("📂 Data Loading")
        
        # Get data directory - simplified for Streamlit Cloud
        data_dir = Path("actual_generation")
        
        if not data_dir.exists():
            st.error(f"Data directory not found at {data_dir}")
            st.info("Please ensure the 'actual_generation' folder is in the correct location")
            return False
            
        # List available files
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            st.error("No CSV files found in the actual_generation directory!")
            return False
            
        # File selection
        file_names = [f.name for f in csv_files]
        selected_file = st.selectbox(
            "Select data file:",
            options=file_names,
            index=file_names.index("Blue_Wing_Solar_Energy_actual_generation.csv") if "Blue_Wing_Solar_Energy_actual_generation.csv" in file_names else 0
        )
        
        # Load button
        if st.button("Load Data", type="primary"):
            filepath = data_dir / selected_file
            try:
                data = pd.read_csv(filepath)
                data['Date'] = pd.to_datetime(data['Date'])
                
                # Detect generation type (Solar or Wind)
                if 'Solar (MWh)' in data.columns:
                    generation_col = 'Solar (MWh)'
                    generation_type = 'Solar'
                elif 'Wind (MWh)' in data.columns:
                    generation_col = 'Wind (MWh)'
                    generation_type = 'Wind'
                else:
                    st.error("No 'Solar (MWh)' or 'Wind (MWh)' column found in the data!")
                    return False
                
                # Create a standardized column name for analysis
                data['Generation (MWh)'] = data[generation_col]
                
                # Check for Month column
                if 'Month' not in data.columns:
                    # Try to extract month from Date if Month column doesn't exist
                    data['Month'] = data['Date'].dt.month
                    st.info("Month column not found, extracted from Date column")
                
                # Store in session state
                st.session_state.data = data
                st.session_state.data_loaded = True
                st.session_state.selected_file = selected_file
                st.session_state.generation_type = generation_type
                st.session_state.generation_col = generation_col
                
                # Try to load site registry and match capacity
                self.load_site_capacity()
                
                st.success(f"✅ Successfully loaded {len(data)} months of {generation_type} data")
                return True
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return False
                
        # Display data info if loaded
        if st.session_state.data_loaded:
            data = st.session_state.data
            gen_type = st.session_state.generation_type
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Months", len(data))
            with col2:
                st.metric("Date Range", f"{data['Date'].min().strftime('%Y-%m')} to {data['Date'].max().strftime('%Y-%m')}")
            with col3:
                st.metric(f"Avg Monthly {gen_type}", f"{data['Generation (MWh)'].mean():,.0f} MWh")
            with col4:
                if st.session_state.facility_capacity:
                    st.metric("AC Capacity", f"{st.session_state.facility_capacity:,.1f} MW")
                else:
                    st.metric(f"Total {gen_type}", f"{data['Generation (MWh)'].sum():,.0f} MWh")
                
            # Show data preview
            with st.expander("📊 Data Preview"):
                display_cols = ['Date', 'Month-Year', st.session_state.generation_col]
                available_cols = [col for col in display_cols if col in data.columns]
                st.dataframe(data[available_cols].head(10))
                
        return st.session_state.data_loaded
        
    def load_site_capacity(self):
        """Load site registry to get facility capacity"""
        try:
            # Try to load site registry from parent directory first
            registry_paths = [
                Path("site_registry.csv"),
                Path("actual_generation/site_registry.csv"),
                Path("../site_registry.csv")
            ]
            
            site_registry = None
            for path in registry_paths:
                if path.exists():
                    site_registry = pd.read_csv(path)
                    break
                    
            if site_registry is not None:
                st.session_state.site_registry = site_registry
                
                # Try to match facility name
                if 'Plant Name' in st.session_state.data.columns:
                    facility_name = st.session_state.data['Plant Name'].iloc[0]
                    
                    # Look for match in registry (assuming columns like 'Facility Name' and 'AC Capacity (MW)')
                    for col in ['Facility Name', 'Plant Name', 'Name']:
                        if col in site_registry.columns:
                            match = site_registry[site_registry[col].str.contains(facility_name.split()[0], case=False, na=False)]
                            if not match.empty:
                                # Look for capacity column
                                for cap_col in ['AC Capacity (MW)', 'Capacity (MW)', 'AC_Capacity_MW', 'Capacity']:
                                    if cap_col in match.columns:
                                        st.session_state.facility_capacity = match[cap_col].iloc[0]
                                        st.info(f"✅ Found facility capacity: {st.session_state.facility_capacity:.1f} MW")
                                        break
                                break
        except Exception as e:
            # If registry not found or error, continue without it
            pass
        
    def analysis_parameters_section(self):
        """Handle analysis parameters selection"""
        st.header("⚙️ Analysis Parameters")
        
        # Add educational content
        with st.expander("📚 Understanding Parametric Insurance", expanded=False):
            st.markdown("""
            **What is Parametric Insurance?**
            
            Unlike traditional insurance that pays for actual damages, parametric insurance pays automatically when a measurable parameter (like energy generation) falls below a pre-agreed threshold.
            
            **Example:**
            - Threshold (VaR): 2,850 MWh
            - Actual generation: 2,700 MWh  
            - Automatic payout: (2,850 - 2,700) × Energy Price
            
            **Benefits:**
            - ✅ Instant payouts (no claims process)
            - ✅ Transparent triggers
            - ✅ No need to prove losses
            """)
        
        data = st.session_state.data
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Analysis start date
            min_date = data['Date'].min().date()
            max_date = data['Date'].max().date()
            
            start_date = st.date_input(
                "Analysis start date:",
                value=pd.to_datetime('2012-01-01').date() if pd.to_datetime('2012-01-01') >= data['Date'].min() else min_date,
                min_value=min_date,
                max_value=max_date,
                help="Choose how far back to analyze. More historical data = better estimates"
            )
            
        with col2:
            # VaR threshold selection
            st.subheader("VaR Threshold")
            
            # Educational info about VaR
            st.info("""
            **VaR (Value at Risk)** = The threshold that triggers insurance payouts
            
            • **P10**: 10% chance of falling below (standard coverage)
            • **P5**: 5% chance of falling below (enhanced coverage)  
            • **P1**: 1% chance of falling below (catastrophic coverage)
            
            Lower percentile = Lower threshold = More frequent payouts = Higher premium
            """)
            
            threshold_option = st.radio(
                "Select threshold type:",
                options=["Standard (P10)", "Enhanced (P5)", "Catastrophic (P1)", "Custom"],
                horizontal=True
            )
            
            if threshold_option == "Custom":
                custom_percentile = st.slider(
                    "Custom percentile:",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1
                )
                threshold_percentile = custom_percentile
            else:
                threshold_map = {
                    "Standard (P10)": 10,
                    "Enhanced (P5)": 5,
                    "Catastrophic (P1)": 1
                }
                threshold_percentile = threshold_map[threshold_option]
                
        # Store in session state
        st.session_state.analysis_start_date = pd.to_datetime(start_date)
        st.session_state.threshold_percentile = threshold_percentile
        
        # Filter data
        filtered_data = data[data['Date'] >= st.session_state.analysis_start_date].copy()
        # Ensure Generation (MWh) column exists
        if 'Generation (MWh)' not in filtered_data.columns and st.session_state.generation_col in filtered_data.columns:
            filtered_data['Generation (MWh)'] = filtered_data[st.session_state.generation_col]
        st.session_state.analysis_data = filtered_data
        
        # Show analysis period info
        st.info(f"📊 Analyzing {len(st.session_state.analysis_data)} months from {start_date.strftime('%Y-%m')} | Using P{threshold_percentile} threshold")
        
    def pricing_parameters_section(self):
        """Handle insurance pricing parameters"""
        st.header("💰 Insurance Pricing Parameters")
        
        # Add educational content about pricing
        with st.expander("📚 How Insurance Pricing Works", expanded=False):
            st.markdown("""
            **Insurance Premium Components:**
            
            1. **Pure Premium** = Expected annual loss (probability × severity)
            2. **Risk Load** = Multiplier for uncertainty (1.2x - 2.0x is industry standard)
            3. **Expenses** = Operating costs (typically 15-25% for renewables)
            4. **Profit Margin** = Target profit (usually 10-15%)
            
            **Coverage Limit Calculation:**
            - Based on facility's annual revenue potential
            - Annual Revenue = Capacity (MW) × Capacity Factor × 8760 hours × Energy Price
            - Coverage typically 80-100% of annual revenue
            
            **Industry Standard Ranges:**
            - **Conservative**: Risk load 1.8-2.0x, expenses 20-25%, profit 12-15%
            - **Standard**: Risk load 1.5x, expenses 18-20%, profit 10-12%
            - **Aggressive**: Risk load 1.2-1.3x, expenses 15-18%, profit 8-10%
            """)
        
        # Pricing scenario selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            pricing_scenario = st.selectbox(
                "Pricing Scenario:",
                options=["Conservative", "Standard", "Aggressive", "Custom"],
                help="Select a predefined pricing scenario or choose Custom to set your own parameters"
            )
        
        # Define more realistic scenario presets
        scenarios = {
            "Conservative": {"risk_load": 1.8, "expense": 22, "profit": 12},
            "Standard": {"risk_load": 1.5, "expense": 20, "profit": 10},
            "Aggressive": {"risk_load": 1.2, "expense": 17, "profit": 8}
        }
        
        # Parameter inputs
        with col2:
            if pricing_scenario == "Custom":
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    risk_load = st.number_input(
                        "Risk Load Factor:",
                        min_value=1.0,
                        max_value=2.5,  # Reduced from 5.0
                        value=1.5,
                        step=0.1,
                        help="Industry standard: 1.2x - 2.0x"
                    )
                with col_b:
                    expense_pct = st.number_input(
                        "Expense Ratio (%):",
                        min_value=10,
                        max_value=30,  # Reduced from 50
                        value=20,
                        step=1,
                        help="Industry standard: 15-25%"
                    )
                with col_c:
                    profit_pct = st.number_input(
                        "Profit Margin (%):",
                        min_value=5,
                        max_value=20,  # Reduced from 50
                        value=10,
                        step=1,
                        help="Industry standard: 8-15%"
                    )
            else:
                # Display preset values
                preset = scenarios[pricing_scenario]
                risk_load = preset["risk_load"]
                expense_pct = preset["expense"]
                profit_pct = preset["profit"]
                
                st.info(f"**{pricing_scenario} Pricing**: Risk Load: {risk_load}x | Expenses: {expense_pct}% | Profit: {profit_pct}%")
        
        # Additional parameters
        st.subheader("Coverage and Market Parameters")
        
        # Calculate suggested coverage limit based on capacity
        if st.session_state.facility_capacity:
            # Calculate potential annual revenue
            st.info(f"🏭 Facility AC Capacity: {st.session_state.facility_capacity:.1f} MW")
            
            col3, col4 = st.columns(2)
            
            with col3:
                energy_price = st.number_input(
                    "Energy Price ($/MWh):",
                    min_value=10,
                    max_value=150,
                    value=50,
                    step=5,
                    help="Market price per MWh"
                )
                
                # Estimate capacity factor based on generation type
                default_cf = 0.25 if st.session_state.generation_type == 'Solar' else 0.35
                capacity_factor = st.slider(
                    "Capacity Factor:",
                    min_value=0.10,
                    max_value=0.50,
                    value=default_cf,
                    step=0.01,
                    help=f"Typical {st.session_state.generation_type}: {'20-30%' if st.session_state.generation_type == 'Solar' else '30-45%'}"
                )
            
            with col4:
                # Calculate annual revenue potential
                annual_generation = st.session_state.facility_capacity * 8760 * capacity_factor
                annual_revenue = annual_generation * energy_price
                
                st.metric("Annual Generation Potential", f"{annual_generation:,.0f} MWh")
                st.metric("Annual Revenue Potential", f"${annual_revenue:,.0f}")
                
            # Coverage limit options
            st.subheader("Coverage Limit")
            
            coverage_option = st.radio(
                "Coverage basis:",
                options=["% of Annual Revenue", "Custom Amount"],
                horizontal=True
            )
            
            if coverage_option == "% of Annual Revenue":
                coverage_pct = st.slider(
                    "Coverage as % of annual revenue:",
                    min_value=50,
                    max_value=100,
                    value=80,
                    step=5,
                    help="Typically 80-100% of annual revenue"
                )
                coverage_limit = annual_revenue * (coverage_pct / 100)
                st.success(f"📊 Coverage Limit: ${coverage_limit:,.0f} ({coverage_pct}% of annual revenue)")
            else:
                coverage_limit = st.number_input(
                    "Custom Coverage Limit ($):",
                    min_value=100_000,
                    max_value=int(annual_revenue * 2),
                    value=int(annual_revenue * 0.8),
                    step=100_000,
                    help=f"Suggested range: ${annual_revenue*0.5:,.0f} - ${annual_revenue*1.2:,.0f}"
                )
        else:
            # No capacity data - use traditional inputs
            col3, col4, col5 = st.columns(3)
            
            with col3:
                energy_price = st.number_input(
                    "Energy Price ($/MWh):",
                    min_value=10,
                    max_value=150,
                    value=50,
                    step=5,
                    help="Market price per MWh"
                )
            
            with col4:
                # If no capacity, estimate from historical generation
                avg_monthly_gen = st.session_state.data['Generation (MWh)'].mean() if st.session_state.data_loaded else 2000
                estimated_annual = avg_monthly_gen * 12
                default_limit = estimated_annual * energy_price * 0.8
                
                coverage_limit = st.number_input(
                    "Coverage Limit ($):",
                    min_value=100_000,
                    max_value=50_000_000,
                    value=min(int(default_limit), 10_000_000),
                    step=100_000,
                    help=f"Based on avg generation: ${default_limit:,.0f} suggested"
                )
                
            with col5:
                confidence_adjustment = st.checkbox(
                    "Auto-adjust for data confidence",
                    value=True,
                    help="Automatically increase risk load for months with limited historical data. Months with only 1 breach get higher risk loading due to uncertainty."
                )
        
        # Store parameters
        st.session_state.risk_load_factor = risk_load
        st.session_state.expense_ratio = expense_pct / 100
        st.session_state.profit_margin = profit_pct / 100
        st.session_state.energy_price = energy_price
        st.session_state.coverage_limit = coverage_limit
        st.session_state.confidence_adjustment = confidence_adjustment if 'confidence_adjustment' in locals() else True
        
    def run_analysis(self):
        """Run the complete analysis"""
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Running analysis..."):
                progress_bar = st.progress(0)
                
                # Step 1: Calculate monthly statistics
                progress_bar.progress(20, "Calculating monthly statistics...")
                self.calculate_monthly_statistics()
                
                # Step 2: Fit distributions
                progress_bar.progress(40, "Fitting distributions...")
                self.fit_distributions()
                
                # Step 3: Auto-select best distributions
                progress_bar.progress(60, "Selecting optimal distributions...")
                self.auto_select_distributions()
                
                # Step 4: Calculate VaR
                progress_bar.progress(80, "Calculating VaR thresholds...")
                self.calculate_var()
                
                # Step 5: Calculate CVaR
                progress_bar.progress(90, "Calculating CVaR and expected losses...")
                self.calculate_cvar()
                
                progress_bar.progress(100, "Analysis complete!")
                st.session_state.analysis_complete = True
                
    def calculate_monthly_statistics(self):
        """Calculate statistics for each calendar month"""
        analysis_data = st.session_state.analysis_data
        monthly_stats = {}
        
        for month in range(1, 13):
            month_data = analysis_data[analysis_data['Month'] == month]['Generation (MWh)'].values
            
            if len(month_data) > 0:
                percentiles = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
                emp_percentiles = {f'P{p}': np.percentile(month_data, p) for p in percentiles}
                
                monthly_stats[month] = {
                    'month_name': pd.to_datetime(f"2000-{month}-01").strftime('%B'),
                    'data': month_data,
                    'count': len(month_data),
                    'mean': np.mean(month_data),
                    'std': np.std(month_data),
                    'min': np.min(month_data),
                    'max': np.max(month_data),
                    'empirical_percentiles': emp_percentiles
                }
                
        st.session_state.monthly_stats = monthly_stats
        
    def fit_distributions(self):
        """Fit distributions to monthly data"""
        distributions = {
            'normal': stats.norm,
            'lognormal': stats.lognorm,
            'gamma': stats.gamma,
            'weibull': stats.weibull_min,
            'beta': stats.beta
        }
        
        distribution_results = {}
        
        for month, data in st.session_state.monthly_stats.items():
            values = data['data']
            
            if len(values) < 5:
                continue
                
            results = {}
            
            for dist_name, dist_func in distributions.items():
                try:
                    if dist_name == 'beta':
                        scaled_values = (values - values.min()) / (values.max() - values.min() + 1e-10)
                        params = dist_func.fit(scaled_values)
                        params = params + (values.min(), values.max())
                        ks_stat, ks_pval = stats.kstest(scaled_values, lambda x: dist_func.cdf(x, *params[:-2]))
                        log_likelihood = np.sum(dist_func.logpdf(scaled_values, *params[:-2]))
                    else:
                        params = dist_func.fit(values)
                        ks_stat, ks_pval = stats.kstest(values, lambda x: dist_func.cdf(x, *params))
                        log_likelihood = np.sum(dist_func.logpdf(values, *params))
                    
                    n_params = len(params)
                    aic = 2 * n_params - 2 * log_likelihood
                    
                    # Calculate P10 error
                    p10_empirical = np.percentile(values, st.session_state.threshold_percentile)
                    if dist_name == 'beta':
                        p10_fitted = params[-2] + (params[-1] - params[-2]) * dist_func.ppf(st.session_state.threshold_percentile/100, *params[:-2])
                    else:
                        p10_fitted = dist_func.ppf(st.session_state.threshold_percentile/100, *params)
                    
                    p10_error = abs(p10_fitted - p10_empirical) / (p10_empirical + 1e-10) * 100
                    
                    results[dist_name] = {
                        'params': params,
                        'ks_stat': ks_stat,
                        'ks_pval': ks_pval,
                        'aic': aic,
                        'p10_error': p10_error,
                        'fit_score': (1 - ks_stat) * 0.4 + (ks_pval) * 0.3 + (1 / (1 + p10_error/10)) * 0.3
                    }
                except:
                    continue
                    
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            if valid_results:
                sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['fit_score'], reverse=True)
                distribution_results[month] = {
                    'all_results': results,
                    'sorted_results': sorted_results,
                    'best_dist': sorted_results[0]
                }
                
        st.session_state.distribution_results = distribution_results
        
    def auto_select_distributions(self):
        """Automatically select best distributions"""
        selected_distributions = {}
        for month, dist_data in st.session_state.distribution_results.items():
            selected_distributions[month] = dist_data['best_dist']
        st.session_state.selected_distributions = selected_distributions
        
    def calculate_var(self):
        """Calculate VaR using fitted distributions"""
        percentile = st.session_state.threshold_percentile
        
        for month in range(1, 13):
            if month in st.session_state.monthly_stats and month in st.session_state.selected_distributions:
                values = st.session_state.monthly_stats[month]['data']
                
                # Empirical percentile
                emp_var = np.percentile(values, percentile)
                
                # Fitted percentile
                dist_name, dist_info = st.session_state.selected_distributions[month]
                params = dist_info['params']
                
                if dist_name == 'normal':
                    fitted_var = stats.norm.ppf(percentile/100, *params)
                elif dist_name == 'lognormal':
                    fitted_var = stats.lognorm.ppf(percentile/100, *params)
                elif dist_name == 'gamma':
                    fitted_var = stats.gamma.ppf(percentile/100, *params)
                elif dist_name == 'weibull':
                    fitted_var = stats.weibull_min.ppf(percentile/100, *params)
                elif dist_name == 'beta':
                    scaled_var = stats.beta.ppf(percentile/100, *params[:-2])
                    fitted_var = params[-2] + (params[-1] - params[-2]) * scaled_var
                
                st.session_state.monthly_stats[month]['var_empirical'] = emp_var
                st.session_state.monthly_stats[month]['var_fitted'] = fitted_var
                st.session_state.monthly_stats[month]['var_distribution'] = dist_name
                
    def calculate_cvar(self):
        """Calculate CVaR using empirical data"""
        cvar_results = {}
        
        for month in range(1, 13):
            if month not in st.session_state.monthly_stats:
                continue
                
            month_name = st.session_state.monthly_stats[month]['month_name']
            month_data_all = st.session_state.analysis_data[st.session_state.analysis_data['Month'] == month].copy()
            
            # Use fitted VaR as threshold
            threshold = st.session_state.monthly_stats[month]['var_fitted']
            
            # Find months below threshold
            below_threshold = month_data_all[month_data_all['Generation (MWh)'] < threshold]
            
            if len(below_threshold) > 0:
                cvar = below_threshold['Generation (MWh)'].mean()
                shortfall = threshold - cvar
                
                cvar_results[month] = {
                    'month_name': month_name,
                    'threshold': threshold,
                    'breach_count': len(below_threshold),
                    'total_months': len(month_data_all),
                    'breach_probability': len(below_threshold) / len(month_data_all) * 100,
                    'cvar': cvar,
                    'average_shortfall': shortfall,
                    'breach_dates': below_threshold['Date'].tolist(),
                    'breach_values': below_threshold['Generation (MWh)'].tolist()
                }
            else:
                cvar_results[month] = {
                    'month_name': month_name,
                    'threshold': threshold,
                    'breach_count': 0,
                    'total_months': len(month_data_all),
                    'breach_probability': 0,
                    'cvar': None,
                    'average_shortfall': 0
                }
                
        st.session_state.cvar_results = cvar_results
        
    def display_results(self):
        """Display analysis results"""
        if not st.session_state.get('analysis_complete', False):
            st.info("👆 Please run the analysis first")
            return
            
        st.header("📊 Analysis Results")
        
        # Add quick glossary
        with st.expander("📖 Quick Glossary", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Key Terms:**
                - **VaR**: Value at Risk - the threshold that triggers payouts
                - **CVaR**: Conditional VaR - average generation when below threshold
                - **Breach**: When generation falls below the VaR threshold
                - **Shortfall**: The difference between threshold and actual generation
                - **Expected Loss**: Probability × Average Shortfall
                """)
                
            with col2:
                st.markdown("""
                **Pricing Terms:**
                - **Pure Premium**: Expected annual loss in dollars
                - **Risk Load**: Safety margin for uncertainty
                - **Rate on Line**: Premium as % of coverage limit
                - **Loss Ratio**: Expected losses as % of premium
                - **Confidence**: How reliable our estimates are
                """)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Summary", "📊 Monthly Details", "🎯 Pricing", "📉 Visualizations"])
        
        with tab1:
            self.display_summary()
            
        with tab2:
            self.display_monthly_details()
            
        with tab3:
            self.display_pricing()
            
        with tab4:
            self.display_visualizations()
            
    def display_summary(self):
        """Display summary statistics"""
        st.subheader("Executive Summary")
        
        # Add CVaR explanation
        with st.expander("📚 Understanding CVaR (Conditional Value at Risk)", expanded=False):
            st.markdown("""
            **CVaR answers: "When things go bad, how bad do they get?"**
            
            While VaR tells us the threshold, CVaR tells us what happens below that threshold.
            
            **Example for December:**
            - VaR (threshold): 2,850 MWh
            - Historical breaches: 2,800 MWh and 2,700 MWh
            - CVaR: (2,800 + 2,700) / 2 = 2,750 MWh
            - Average shortfall: 2,850 - 2,750 = 100 MWh
            
            **Why it matters:**
            - VaR tells us WHEN to pay (the trigger)
            - CVaR tells us HOW MUCH to pay (average payout)
            - Expected Loss = Breach Probability × Average Shortfall
            """)
            
            # Visual representation
            st.markdown("""
            ```
            Generation Distribution:
                 |
            3500 |    ████████
            3000 |  ████████████
            2500 |████████████████ ← CVaR (average of tail)
            2000 |██████████████|← VaR (threshold)
                 |______________|
                       ↑
                  Payouts occur here
            ```
            """)
        
        # Calculate overall statistics
        total_expected_loss = 0
        high_confidence_months = 0
        low_confidence_months = 0
        
        for month, result in st.session_state.cvar_results.items():
            if result['breach_count'] > 0:
                expected_loss = (result['breach_probability'] / 100) * result['average_shortfall']
                total_expected_loss += expected_loss
                
                if result['breach_count'] >= 3:
                    high_confidence_months += 1
                elif result['breach_count'] == 1:
                    low_confidence_months += 1
                    
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Expected Annual Loss",
                f"{total_expected_loss * 12:,.0f} MWh",
                f"${total_expected_loss * 12 * st.session_state.energy_price:,.0f}",
                help="Total expected shortfall across all months × 12. This is the average annual payout the insurer expects to make."
            )
            
        with col2:
            avg_breach_prob = np.mean([r['breach_probability'] for r in st.session_state.cvar_results.values()])
            st.metric(
                "Avg Breach Probability",
                f"{avg_breach_prob:.1f}%",
                "Across all months",
                help="Average likelihood of generation falling below threshold. Higher % = more frequent payouts."
            )
            
        with col3:
            st.metric(
                "High Confidence Months",
                high_confidence_months,
                "3+ historical breaches",
                help="Months with 3+ historical breaches have reliable CVaR estimates. More is better for pricing accuracy."
            )
            
        with col4:
            st.metric(
                "Low Confidence Months",
                low_confidence_months,
                "Only 1 breach",
                help="Months with only 1 breach have uncertain estimates. These require higher risk loads."
            )
            
        # Add capacity factor analysis if we have capacity data OR estimate it
        st.markdown("---")
        st.subheader("Facility Performance Analysis")
        
        if st.session_state.facility_capacity:
            # Calculate actual capacity factor
            total_generation = st.session_state.analysis_data['Generation (MWh)'].sum()
            total_months = len(st.session_state.analysis_data)
            total_hours = total_months * 730  # Average hours per month
            
            actual_capacity_factor = total_generation / (st.session_state.facility_capacity * total_hours)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Facility Capacity",
                    f"{st.session_state.facility_capacity:.1f} MW",
                    "AC Nameplate"
                )
                
            with col2:
                st.metric(
                    "Historical Capacity Factor",
                    f"{actual_capacity_factor:.1%}",
                    f"Over {total_months} months"
                )
                
            with col3:
                monthly_avg = total_generation / total_months
                theoretical_monthly = st.session_state.facility_capacity * 730 * actual_capacity_factor
                st.metric(
                    "Avg Monthly Generation",
                    f"{monthly_avg:,.0f} MWh",
                    f"{(monthly_avg/theoretical_monthly - 1)*100:+.1f}% vs expected"
                )
        else:
            # Estimate capacity from generation data
            total_generation = st.session_state.analysis_data['Generation (MWh)'].sum()
            total_months = len(st.session_state.analysis_data)
            monthly_avg = total_generation / total_months
            
            # Estimate capacity assuming typical capacity factors
            assumed_cf = 0.25 if st.session_state.generation_type == 'Solar' else 0.35
            estimated_capacity = monthly_avg / (730 * assumed_cf)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Estimated Capacity",
                    f"~{estimated_capacity:.0f} MW",
                    f"Assuming {assumed_cf:.0%} CF",
                    help="Estimated from generation data"
                )
                
            with col2:
                st.metric(
                    "Avg Monthly Generation",
                    f"{monthly_avg:,.0f} MWh",
                    f"Over {total_months} months"
                )
                
            with col3:
                annual_avg = monthly_avg * 12
                st.metric(
                    "Annual Generation",
                    f"{annual_avg:,.0f} MWh/yr",
                    f"${annual_avg * st.session_state.energy_price:,.0f}/yr"
                )
            
    def display_monthly_details(self):
        """Display detailed monthly results"""
        st.subheader("Monthly VaR and CVaR Details")
        
        # Add explanation of the table
        st.info("""
        **How to read this table:**
        - **VaR**: The threshold below which insurance pays out
        - **Breaches**: How many times historically generation fell below VaR
        - **Breach %**: Probability of needing to pay out
        - **CVaR**: Average generation when below threshold
        - **Avg Shortfall**: Average payout amount when triggered
        """)
        
        # Create DataFrame for display
        results_data = []
        for month in range(1, 13):
            if month in st.session_state.monthly_stats and month in st.session_state.cvar_results:
                stats = st.session_state.monthly_stats[month]
                cvar = st.session_state.cvar_results[month]
                
                results_data.append({
                    'Month': stats['month_name'],
                    'Data Points': stats['count'],
                    'Mean (MWh)': f"{stats['mean']:,.0f}",
                    'VaR (MWh)': f"{stats.get('var_fitted', 0):,.0f}",
                    'Distribution': stats.get('var_distribution', 'N/A'),
                    'Breaches': cvar['breach_count'],
                    'Breach %': f"{cvar['breach_probability']:.1f}%",
                    'CVaR (MWh)': f"{cvar['cvar']:,.0f}" if cvar['cvar'] else "N/A",
                    'Avg Shortfall': f"{cvar['average_shortfall']:,.0f}"
                })
                
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True, hide_index=True)
        
        # Detailed breach information
        with st.expander("📋 View Breach Details"):
            for month, result in st.session_state.cvar_results.items():
                if result['breach_count'] > 0:
                    st.write(f"**{result['month_name']}** - {result['breach_count']} breaches:")
                    breach_df = pd.DataFrame({
                        'Date': pd.to_datetime(result['breach_dates']).strftime('%Y-%m'),
                        'Generation (MWh)': result['breach_values'],
                        'Shortfall (MWh)': [result['threshold'] - v for v in result['breach_values']]
                    })
                    st.dataframe(breach_df, use_container_width=True, hide_index=True)
                    
    def display_pricing(self):
        """Display insurance pricing calculations"""
        st.subheader("💰 Insurance Pricing Calculation")
        
        # Add visual explanation of pricing flow
        with st.expander("🎯 How We Calculate Your Premium", expanded=True):
            st.markdown("""
            ```
            Expected Annual Loss (MWh) × Energy Price ($)
                                ↓
            Pure Premium (Expected Loss in $)
                                ↓
                        × Risk Load Factor
                                ↓
            Risk-Loaded Premium
                                ↓
                        + Expenses (%)
                                ↓
            Subtotal
                                ↓
                        + Profit Margin (%)
                                ↓
            Final Annual Premium 💰
            ```
            """)
        
        # Calculate total expected loss and confidence metrics
        total_expected_loss = 0
        high_confidence_months = 0
        low_confidence_months = 0
        no_breach_months = 0
        
        for month, result in st.session_state.cvar_results.items():
            if result['breach_count'] > 0:
                expected_loss = (result['breach_probability'] / 100) * result['average_shortfall']
                total_expected_loss += expected_loss
                
                if result['breach_count'] >= 3:
                    high_confidence_months += 1
                elif result['breach_count'] == 1:
                    low_confidence_months += 1
            else:
                no_breach_months += 1
                
        annual_expected_loss = total_expected_loss * 12
        energy_price = st.session_state.energy_price
        
        # Determine risk load based on confidence
        if st.session_state.confidence_adjustment:
            low_confidence_months = sum(1 for r in st.session_state.cvar_results.values() if r['breach_count'] == 1)
            no_breach_months = sum(1 for r in st.session_state.cvar_results.values() if r['breach_count'] == 0)
            
            # More nuanced risk adjustment
            if no_breach_months >= 6:
                actual_risk_load = st.session_state.risk_load_factor * 1.3
                st.warning(f"""
                ⚠️ **Risk load increased to {actual_risk_load:.2f}x due to limited data**
                
                {no_breach_months} months have NO historical breaches, making CVaR estimation impossible.
                Consider using a lower threshold (P5 instead of P10) for better data coverage.
                """)
            elif low_confidence_months > 6:
                actual_risk_load = st.session_state.risk_load_factor * 1.15
                st.warning(f"""
                ⚠️ **Risk load increased to {actual_risk_load:.2f}x due to low data confidence**
                
                {low_confidence_months} months have only 1 historical breach, making estimates uncertain. 
                The additional risk load protects against underpricing due to limited data.
                """)
            elif low_confidence_months > 3:
                actual_risk_load = st.session_state.risk_load_factor * 1.1
                st.info(f"""
                ℹ️ **Risk load adjusted to {actual_risk_load:.2f}x for moderate confidence**
                
                {low_confidence_months} months have limited breach data.
                Small risk adjustment applied for prudent pricing.
                """)
            else:
                actual_risk_load = st.session_state.risk_load_factor
                if high_confidence_months >= 9:
                    st.success(f"""
                    ✅ **High confidence pricing**: Risk load {actual_risk_load:.1f}x
                    
                    {high_confidence_months} months have 3+ breaches, providing reliable CVaR estimates.
                    """)
        else:
            actual_risk_load = st.session_state.risk_load_factor
            
        # Calculate pricing components
        pure_premium = annual_expected_loss * energy_price
        risk_loaded_premium = pure_premium * actual_risk_load
        expenses = risk_loaded_premium * st.session_state.expense_ratio
        subtotal = risk_loaded_premium + expenses
        profit = subtotal * st.session_state.profit_margin
        total_premium = subtotal + profit
        
        # Display pricing breakdown
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Pricing Components")
            
            # Create pricing table with color coding
            pricing_data = [
                {"Component": "1. Pure Premium (Expected Loss)", "Calculation": f"{annual_expected_loss:,.0f} MWh × ${energy_price}", "Amount": f"${pure_premium:,.0f}"},
                {"Component": f"2. Risk Load ({actual_risk_load:.1f}x)", "Calculation": f"${pure_premium:,.0f} × {actual_risk_load:.1f}", "Amount": f"${risk_loaded_premium:,.0f}"},
                {"Component": f"3. Expenses ({st.session_state.expense_ratio*100:.0f}%)", "Calculation": f"${risk_loaded_premium:,.0f} × {st.session_state.expense_ratio:.0%}", "Amount": f"${expenses:,.0f}"},
                {"Component": f"4. Profit Margin ({st.session_state.profit_margin*100:.0f}%)", "Calculation": f"${subtotal:,.0f} × {st.session_state.profit_margin:.0%}", "Amount": f"${profit:,.0f}"}
            ]
            
            pricing_df = pd.DataFrame(pricing_data)
            st.dataframe(pricing_df, use_container_width=True, hide_index=True)
            
            # Highlight final premium
            st.success(f"### 💰 TOTAL ANNUAL PREMIUM: ${total_premium:,.0f}")
            
            # Monthly breakdown
            st.info(f"**Monthly Premium**: ${total_premium/12:,.0f} | **Daily Premium**: ${total_premium/365:,.0f}")
            
        with col2:
            st.markdown("### Key Metrics")
            
            # Rate on line
            rate_on_line = (total_premium / st.session_state.coverage_limit) * 100
            st.metric("Rate on Line", f"{rate_on_line:.2f}%", 
                     help="Premium as % of limit. Industry standard: 2-10% for renewable energy")
            
            # Loss ratio
            expected_loss_ratio = (pure_premium / total_premium) * 100
            st.metric("Expected Loss Ratio", f"{expected_loss_ratio:.1f}%",
                     help="Expected payouts as % of premium. Target: 40-60%")
            
            # Monthly premium
            st.metric("Monthly Premium", f"${total_premium/12:,.0f}")
            
            # Add industry benchmark comparison
            if rate_on_line < 2:
                st.warning("⚠️ Rate on Line below 2% - may be underpriced")
            elif rate_on_line > 10:
                st.warning("⚠️ Rate on Line above 10% - may be uncompetitive")
            else:
                st.success("✅ Rate on Line within industry norms")
                
        # Additional pricing insights
        st.markdown("---")
        st.subheader("💡 Pricing Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Show breakdown as percentages
            st.markdown("**Cost Structure Breakdown:**")
            breakdown_df = pd.DataFrame([
                {"Component": "Expected Losses", "% of Premium": f"{(pure_premium/total_premium)*100:.1f}%"},
                {"Component": "Risk Margin", "% of Premium": f"{((risk_loaded_premium-pure_premium)/total_premium)*100:.1f}%"},
                {"Component": "Operating Expenses", "% of Premium": f"{(expenses/total_premium)*100:.1f}%"},
                {"Component": "Profit Margin", "% of Premium": f"{(profit/total_premium)*100:.1f}%"}
            ])
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
            
        with col2:
            # Coverage adequacy check
            st.markdown("**Coverage Adequacy Analysis:**")
            
            max_monthly_loss = max(r['average_shortfall'] for r in st.session_state.cvar_results.values() if r['breach_count'] > 0)
            max_annual_exposure = max_monthly_loss * 12 * energy_price
            coverage_adequacy = (st.session_state.coverage_limit / max_annual_exposure) * 100
            
            if coverage_adequacy < 100:
                st.error(f"⚠️ Coverage may be insufficient: {coverage_adequacy:.0f}% of max exposure")
            else:
                st.success(f"✅ Coverage adequate: {coverage_adequacy:.0f}% of max exposure")
                
            st.info(f"""
            **Coverage Analysis:**
            - Max monthly shortfall: {max_monthly_loss:,.0f} MWh
            - Max annual exposure: ${max_annual_exposure:,.0f}
            - Current limit: ${st.session_state.coverage_limit:,.0f}
            - Coverage ratio: {coverage_adequacy:.0f}%
            """)
            
        # Download pricing report
        if st.button("📥 Download Pricing Report"):
            self.generate_pricing_report(total_premium, rate_on_line, actual_risk_load)
            
    def display_visualizations(self):
        """Display analysis visualizations"""
        st.subheader("📉 Visual Analysis")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Monthly VaR comparison
        ax1 = axes[0, 0]
        months = []
        empirical_vars = []
        fitted_vars = []
        
        for month in range(1, 13):
            if month in st.session_state.monthly_stats and 'var_empirical' in st.session_state.monthly_stats[month]:
                months.append(st.session_state.monthly_stats[month]['month_name'][:3])
                empirical_vars.append(st.session_state.monthly_stats[month]['var_empirical'])
                fitted_vars.append(st.session_state.monthly_stats[month]['var_fitted'])
        
        x = np.arange(len(months))
        width = 0.35
        
        ax1.bar(x - width/2, empirical_vars, width, label='Empirical', alpha=0.7, color='skyblue')
        ax1.bar(x + width/2, fitted_vars, width, label='Fitted', alpha=0.7, color='lightcoral')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Generation (MWh)')
        ax1.set_title(f'Monthly VaR Comparison (P{st.session_state.threshold_percentile})')
        ax1.set_xticks(x)
        ax1.set_xticklabels(months)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Breach Probability by Month
        ax2 = axes[0, 1]
        breach_probs = []
        for month in range(1, 13):
            if month in st.session_state.cvar_results:
                breach_probs.append(st.session_state.cvar_results[month]['breach_probability'])
        
        ax2.plot(months, breach_probs, 'o-', linewidth=2, markersize=8, color='darkgreen')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Breach Probability (%)')
        ax2.set_title('Historical Breach Probability by Month')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        
        # 3. Expected Loss by Month
        ax3 = axes[1, 0]
        expected_losses = []
        for month in range(1, 13):
            if month in st.session_state.cvar_results:
                result = st.session_state.cvar_results[month]
                if result['breach_count'] > 0:
                    expected_loss = (result['breach_probability'] / 100) * result['average_shortfall']
                else:
                    expected_loss = 0
                expected_losses.append(expected_loss)
        
        ax3.bar(months, expected_losses, alpha=0.7, color='orange')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Expected Loss (MWh)')
        ax3.set_title('Expected Monthly Loss')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Distribution Example (December)
        ax4 = axes[1, 1]
        sample_month = 12
        if sample_month in st.session_state.monthly_stats:
            values = st.session_state.monthly_stats[sample_month]['data']
            ax4.hist(values, bins=min(10, len(values)), alpha=0.7, density=True, 
                    edgecolor='black', color='lightblue')
            
            if 'var_fitted' in st.session_state.monthly_stats[sample_month]:
                var_line = st.session_state.monthly_stats[sample_month]['var_fitted']
                ax4.axvline(var_line, color='green', linestyle='--', linewidth=2, 
                           label=f'VaR (P{st.session_state.threshold_percentile})')
            
            ax4.set_xlabel('Generation (MWh)')
            ax4.set_ylabel('Density')
            ax4.set_title(f'December {st.session_state.generation_type} Distribution with Threshold')
            ax4.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
    def generate_pricing_report(self, total_premium, rate_on_line, actual_risk_load):
        """Generate downloadable pricing report"""
        gen_type = st.session_state.get('generation_type', 'Renewable Energy')
        report = f"""
{gen_type.upper()} GENERATION PARAMETRIC INSURANCE PRICING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

FACILITY INFORMATION
File: {st.session_state.selected_file}
Generation Type: {gen_type}
Analysis Period: {st.session_state.analysis_start_date.strftime('%Y-%m')} to {st.session_state.data['Date'].max().strftime('%Y-%m')}
Total Months Analyzed: {len(st.session_state.analysis_data)}

COVERAGE PARAMETERS
Threshold: P{st.session_state.threshold_percentile} ({st.session_state.threshold_percentile}th percentile)
Coverage Limit: ${st.session_state.coverage_limit:,.0f}
Energy Price: ${st.session_state.energy_price}/MWh

PRICING PARAMETERS
Risk Load Factor: {actual_risk_load:.2f}x
Expense Ratio: {st.session_state.expense_ratio:.0%}
Profit Margin: {st.session_state.profit_margin:.0%}

FINAL PRICING
Annual Premium: ${total_premium:,.0f}
Monthly Premium: ${total_premium/12:,.0f}
Rate on Line: {rate_on_line:.2f}%

MONTHLY BREAKDOWN
"""
        
        for month in range(1, 13):
            if month in st.session_state.cvar_results:
                result = st.session_state.cvar_results[month]
                report += f"\n{result['month_name']}:"
                report += f"\n  VaR Threshold: {result['threshold']:,.0f} MWh"
                report += f"\n  Breach Probability: {result['breach_probability']:.1f}%"
                report += f"\n  Average Shortfall: {result['average_shortfall']:,.0f} MWh"
        
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"insurance_pricing_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )
        
    def run(self):
        """Main app runner"""
        # Dynamic title based on loaded data
        if st.session_state.data_loaded and 'generation_type' in st.session_state:
            gen_type = st.session_state.generation_type
            icon = "☀️" if gen_type == "Solar" else "💨"
            st.title(f"{icon} {gen_type} Generation Parametric Insurance Pricing Tool")
        else:
            st.title("⚡ Renewable Energy Parametric Insurance Pricing Tool")
        
        # Welcome message for first-time users
        if not st.session_state.data_loaded:
            st.markdown("""
            ### Welcome! This tool helps you price parametric insurance for renewable energy facilities.
            
            **🎯 What is Parametric Insurance?**
            
            Unlike traditional insurance that requires damage assessment, parametric insurance pays automatically when 
            generation falls below a pre-agreed threshold. Perfect for renewable energy because:
            - ⚡ **Instant payouts** - No claims process needed
            - 📊 **Objective triggers** - Based on actual generation data
            - 💰 **Revenue protection** - Covers lost income from low generation
            
            **📈 How It Works:**
            1. Set a generation threshold (e.g., 10th percentile of historical data)
            2. If monthly generation < threshold → Automatic payout
            3. Payout = (Threshold - Actual) × Energy Price
            
            **🏢 Perfect for:**
            - **Project Developers**: Secure financing with revenue guarantees
            - **Asset Owners**: Protect against weather-related generation losses  
            - **Lenders**: Reduce project risk with coverage for debt service
            - **Insurers**: Price products using advanced statistical methods
            
            **🚀 This Tool Provides:**
            - Industry-standard VaR/CVaR analysis
            - Automatic capacity-based coverage limits
            - Confidence-adjusted risk pricing
            - Professional reports for underwriting
            
            👉 **Get started by loading your generation data below!**
            """)
            
        st.markdown("---")
        
        # Sidebar for navigation
        with st.sidebar:
            st.header("Navigation")
            steps = ["📂 Data Loading", "⚙️ Parameters", "🚀 Analysis", "📊 Results"]
            
            for i, step in enumerate(steps):
                if i == 0:
                    st.markdown(f"**{step}**" if not st.session_state.data_loaded else f"✅ {step}")
                elif i == 1:
                    st.markdown(f"**{step}**" if st.session_state.data_loaded else f"⏳ {step}")
                else:
                    st.markdown(f"**{step}**")
            
            # Add help section in sidebar
            st.markdown("---")
            st.header("❓ Need Help?")
            
            with st.expander("Quick Start Guide"):
                st.markdown("""
                1. **Load Data**: Select your CSV file with monthly generation data
                2. **Set Parameters**: Choose VaR threshold and pricing scenario
                3. **Run Analysis**: Click the button to calculate VaR and CVaR
                4. **Review Results**: Check summary, details, and final pricing
                
                **Tips:**
                - Lower percentiles (P1, P5) = more coverage but higher premiums
                - Conservative pricing = safer but more expensive
                - More historical data = better estimates
                """)
            
            with st.expander("Understanding the Math"):
                st.markdown("""
                **VaR (Value at Risk)**
                - P10 means 10% probability of falling below
                - This becomes your insurance trigger
                
                **CVaR (Conditional VaR)**
                - Average of all values below VaR
                - Tells us the typical payout amount
                
                **Expected Loss**
                - Breach Probability × Average Shortfall
                - Forms the base of your premium
                """)
                
            with st.expander("Industry Standards"):
                st.markdown("""
                **Typical Ranges:**
                - Risk Load: 1.2x - 2.0x
                - Expenses: 15% - 25%
                - Profit: 8% - 15%
                - Rate on Line: 2% - 10%
                
                **Capacity Factors:**
                - Solar: 20% - 30%
                - Wind: 30% - 45%
                """)
                    
        # Main content
        if not st.session_state.data_loaded:
            self.load_data_section()
        else:
            # Show data info at top
            st.success(f"✅ Data loaded: {st.session_state.selected_file}")
            
            # Parameters section
            self.analysis_parameters_section()
            st.markdown("---")
            
            self.pricing_parameters_section()
            st.markdown("---")
            
            # Analysis section
            self.run_analysis()
            st.markdown("---")
            
            # Results section
            self.display_results()

# Run the app
if __name__ == "__main__":
    app = SolarInsurancePricingApp()
    app.run()
