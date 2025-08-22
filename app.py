import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import calendar

# Set page config
st.set_page_config(
    page_title="Clarus Coffee Analytics",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Clarus coffee shop styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #8B4513;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .clarus-brand {
        font-size: 1.4rem;
        color: #D2691E;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #8B4513;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .coffee-specific {
        background: linear-gradient(135deg, #8B4513 0%, #D2691E 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .sidebar .sidebar-content {
        background-color: #FFF8DC;
    }
    .upload-section {
        border: 3px dashed #8B4513;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        background-color: #FFF8DC;
        margin-bottom: 1rem;
    }
    .clarus-footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #8B4513 0%, #D2691E 100%);
        color: white;
        border-radius: 12px;
        margin-top: 3rem;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .stMetric > div > div > div > div {
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    .stMetric > div > div > div > div:first-child {
        font-size: 1.2rem !important;
        color: #8B4513 !important;
        font-weight: 600 !important;
    }
    /* Improve chart readability */
    .js-plotly-plot {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path_or_buffer):
    """Load and preprocess the coffee shop POS data"""
    try:
        # Handle both file paths and uploaded files
        if isinstance(file_path_or_buffer, str):
            # It's a file path
            if file_path_or_buffer.endswith('.xlsx'):
                df = pd.read_excel(file_path_or_buffer)
            else:
                df = pd.read_csv(file_path_or_buffer)
        else:
            # It's an uploaded file buffer
            if file_path_or_buffer.name.endswith('.xlsx'):
                df = pd.read_excel(file_path_or_buffer)
            else:
                df = pd.read_csv(file_path_or_buffer)
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Create additional date columns for easier filtering
        df['Date'] = df['Timestamp'].dt.date
        df['Month'] = df['Timestamp'].dt.to_period('M')
        df['Year'] = df['Timestamp'].dt.year
        df['Day_of_Week'] = df['Timestamp'].dt.day_name()
        df['Hour'] = df['Timestamp'].dt.hour
        df['Month_Name'] = df['Timestamp'].dt.strftime('%B %Y')
        
        # Calculate metrics
        df['Revenue'] = df['Total_Amount']
        df['Discount_Amount'] = df['Quantity'] * df['Unit_Price'] * (df['Discount'] / 100)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure your file has the required columns: Transaction_ID, Timestamp, Employee_ID, Customer_ID, Item_Name, Category, Quantity, Unit_Price, Discount, Payment_Method, Total_Amount, Tax, Tip")
        return None

def create_daily_sales_trend(df):
    """Create daily sales trend line chart for coffee shop with improved readability"""
    daily_sales = df.groupby('Date')['Revenue'].sum().reset_index()
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    
    fig = px.line(
        daily_sales, 
        x='Date', 
        y='Revenue',
        title='Daily Coffee Shop Sales Trend',
        labels={'Revenue': 'Total Revenue ($)', 'Date': 'Date'},
        line_shape='spline'
    )
    
    fig.update_traces(
        line_color='#8B4513',
        line_width=4,
        hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:,.2f}<extra></extra>',
        mode='lines+markers',
        marker=dict(size=6, color='#8B4513')
    )
    
    fig.update_layout(
        title_font_size=24,
        title_x=0.5,
        title_y=0.95,
        title_font_color='#8B4513',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        font_size=14,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(t=70, b=50, l=50, r=50),
        yaxis=dict(
            tickformat='$,.0f',
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        xaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
    )
    
    return fig

def create_category_breakdown(df):
    """Create coffee shop category breakdown chart with improved readability"""
    category_revenue = df.groupby('Category')['Revenue'].sum().reset_index()
    category_revenue = category_revenue.sort_values('Revenue', ascending=False)
    
    # Improved coffee shop specific colors with better contrast
    coffee_colors = ['#8B4513', '#D2691E', '#F4A460', '#CD853F', '#A0522D', '#654321', '#DEB887']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            category_revenue,
            values='Revenue',
            names='Category',
            title='Revenue by Coffee Shop Category',
            color_discrete_sequence=coffee_colors
        )
        fig_pie.update_traces(
            hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>',
            textposition="auto",
            textinfo="percent+label",
            textfont_size=12,
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        fig_pie.update_layout(
            title_font_size=18, 
            title_x=0.5,
            title_y=0.95,
            title_font_color='#8B4513',
            font_size=12,
            height=400,
            margin=dict(t=60, b=40, l=40, r=40)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(
            category_revenue,
            x='Category',
            y='Revenue',
            title='Revenue by Category (Bar Chart)',
            color='Revenue',
            color_continuous_scale=['#F4A460', '#8B4513'],
            text='Revenue'
        )
        fig_bar.update_traces(
            hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>',
            texttemplate='$%{text:,.0f}',
            textposition='outside',
            textfont_size=12
        )
        fig_bar.update_layout(
            title_font_size=18, 
            title_x=0.5,
            title_y=0.95,
            title_font_color='#8B4513',
            xaxis_tickangle=-45,
            font_size=12,
            height=400,
            margin=dict(t=60, b=50, l=50, r=50),
            yaxis=dict(
                tickformat='$,.0f',
                title_font_size=14
            ),
            xaxis=dict(
                title_font_size=14
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True)

def create_coffee_specific_metrics(df):
    """Create coffee shop specific metrics"""
    st.markdown("""
    <div class="coffee-specific">
        <h3>Coffee Shop Insights</h3>
        <p>Key performance indicators specific to your coffee business</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Peak hours analysis
        hourly_sales = df.groupby('Hour')['Revenue'].sum()
        peak_hour = hourly_sales.idxmax()
        peak_revenue = hourly_sales[peak_hour]
        st.metric("Peak Sales Hour", f"{peak_hour}:00", f"${peak_revenue:,.0f}")
    
    with col2:
        # Average items per transaction (basket size)
        items_per_transaction = df.groupby('Transaction_ID')['Quantity'].sum().mean()
        st.metric("Avg Items/Order", f"{items_per_transaction:.1f}")
    
    with col3:
        # Customer frequency
        customer_visits = df.groupby('Customer_ID').size()
        avg_visits = customer_visits.mean()
        st.metric("Avg Customer Visits", f"{avg_visits:.1f}")
    
    with col4:
        # Morning rush percentage (6 AM - 10 AM)
        morning_sales = df[(df['Hour'] >= 6) & (df['Hour'] <= 10)]['Revenue'].sum()
        total_sales = df['Revenue'].sum()
        morning_pct = (morning_sales / total_sales) * 100
        st.metric("Morning Rush %", f"{morning_pct:.1f}%")

def create_hourly_heatmap(df):
    """Create hourly sales heatmap for coffee shop with better readability"""
    hourly_daily = df.groupby(['Day_of_Week', 'Hour'])['Revenue'].sum().reset_index()
    
    # Create pivot table for heatmap
    heatmap_data = hourly_daily.pivot(index='Day_of_Week', columns='Hour', values='Revenue').fillna(0)
    
    # Reorder days properly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    
    # Create custom colorscale for better readability
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Revenue ($)"),
        title="Coffee Shop Sales Heatmap - Revenue by Hour & Day",
        color_continuous_scale=[
            [0.0, '#FFFFFF'],      # White for lowest
            [0.2, '#FFF8DC'],      # Cream
            [0.4, '#F4A460'],      # Sandy brown
            [0.6, '#D2691E'],      # Chocolate
            [0.8, '#8B4513'],      # Saddle brown
            [1.0, '#654321']       # Dark brown for highest
        ],
        aspect="auto"
    )
    
    # Improve layout and readability
    fig.update_layout(
        title_font_size=24,
        title_x=0.5,
        title_y=0.95,
        title_font_color='#8B4513',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        font_size=14,
        height=400,
        margin=dict(t=80, b=50, l=100, r=50),
        coloraxis_colorbar=dict(
            title_font_size=14,
            tickfont_size=12
        )
    )
    
    # Add value annotations for better readability
    fig.update_traces(
        texttemplate="$%{z:,.0f}",
        textfont_size=10,
        textfont_color="black",
        hovertemplate='<b>%{y}</b><br>Hour: %{x}:00<br>Revenue: $%{z:,.2f}<extra></extra>'
    )
    
    return fig

def create_top_products(df):
    """Create top 10 coffee products horizontal bar chart with improved readability"""
    top_products = df.groupby('Item_Name')['Revenue'].sum().reset_index()
    top_products = top_products.sort_values('Revenue', ascending=True).tail(10)
    
    fig = px.bar(
        top_products,
        x='Revenue',
        y='Item_Name',
        orientation='h',
        title='Top 10 Coffee Shop Products by Revenue',
        color='Revenue',
        color_continuous_scale=['#F4A460', '#8B4513'],
        labels={'Revenue': 'Total Revenue ($)', 'Item_Name': 'Product'},
        text='Revenue'
    )
    
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.2f}<extra></extra>',
        texttemplate='$%{text:,.0f}',
        textposition='outside',
        textfont_size=11
    )
    
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        title_y=0.95,
        title_font_color='#8B4513',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        font_size=12,
        height=500,
        margin=dict(t=70, b=50, l=200, r=50),
        xaxis=dict(
            tickformat='$,.0f',
            title='Total Revenue ($)'
        ),
        yaxis=dict(
            title='Product Name'
        )
    )
    
    return fig

def create_payment_method_chart(df):
    """Create payment method distribution donut chart with improved readability"""
    payment_dist = df.groupby('Payment_Method')['Revenue'].sum().reset_index()
    
    # Better color scheme for payment methods
    payment_colors = ['#8B4513', '#D2691E', '#F4A460', '#CD853F', '#A0522D']
    
    fig = go.Figure(data=[go.Pie(
        labels=payment_dist['Payment_Method'],
        values=payment_dist['Revenue'],
        hole=0.4,
        hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>',
        marker_colors=payment_colors,
        textinfo="label+percent",
        textposition="auto",
        textfont_size=12,
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )])
    
    fig.update_layout(
        title='Payment Method Distribution',
        title_font_size=20,
        title_x=0.5,
        title_y=0.92,
        title_font_color='#8B4513',
        font_size=12,
        height=400,
        margin=dict(t=70, b=50, l=50, r=50),
        annotations=[dict(
            text='Payment<br>Methods', 
            x=0.5, y=0.5, 
            font_size=16, 
            font_color='#8B4513',
            showarrow=False
        )]
    )
    
    return fig

def create_employee_performance(df):
    """Create employee performance metrics for coffee shop"""
    emp_metrics = df.groupby('Employee_ID').agg({
        'Revenue': 'sum',
        'Transaction_ID': 'count',
        'Quantity': 'sum'
    }).reset_index()
    
    emp_metrics['Avg_Transaction_Value'] = emp_metrics['Revenue'] / emp_metrics['Transaction_ID']
    emp_metrics['Items_Per_Hour'] = emp_metrics['Quantity'] / 8  # Assuming 8-hour shifts
    emp_metrics = emp_metrics.sort_values('Revenue', ascending=False).head(10)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Employee Revenue Performance', 'Average Transaction Value'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Revenue bar chart
    fig.add_trace(
        go.Bar(
            x=emp_metrics['Employee_ID'],
            y=emp_metrics['Revenue'],
            name='Total Revenue',
            marker_color='#8B4513',
            hovertemplate='<b>Employee %{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Average transaction value
    fig.add_trace(
        go.Bar(
            x=emp_metrics['Employee_ID'],
            y=emp_metrics['Avg_Transaction_Value'],
            name='Avg Transaction Value',
            marker_color='#D2691E',
            hovertemplate='<b>Employee %{x}</b><br>Avg Transaction: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text='Barista Performance Dashboard',
        title_font_size=20,
        title_x=0.5,
        title_y=0.95,
        showlegend=False,
        height=400,
        margin=dict(t=80, b=50, l=80, r=50)
    )
    
    return fig

def create_customer_frequency(df):
    """Create customer visit frequency histogram"""
    customer_visits = df.groupby('Customer_ID').size().reset_index(name='Visit_Count')
    
    fig = px.histogram(
        customer_visits,
        x='Visit_Count',
        nbins=20,
        title='Customer Visit Frequency Distribution',
        labels={'Visit_Count': 'Number of Visits', 'count': 'Number of Customers'},
        color_discrete_sequence=['#8B4513']
    )
    
    fig.update_traces(
        hovertemplate='<b>Visits:</b> %{x}<br><b>Customers:</b> %{y}<extra></extra>'
    )
    
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        title_y=0.92,
        title_font_color='#8B4513',
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        margin=dict(t=80, b=50, l=50, r=50)
    )
    
    return fig

def create_discount_impact(df):
    """Create discount impact analysis for coffee shop"""
    daily_metrics = df.groupby('Date').agg({
        'Revenue': 'sum',
        'Discount_Amount': 'sum'
    }).reset_index()
    daily_metrics['Date'] = pd.to_datetime(daily_metrics['Date'])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Revenue line
    fig.add_trace(
        go.Scatter(
            x=daily_metrics['Date'],
            y=daily_metrics['Revenue'],
            mode='lines+markers',
            name='Daily Revenue',
            line=dict(color='#8B4513', width=3),
            hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:,.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Discount line
    fig.add_trace(
        go.Scatter(
            x=daily_metrics['Date'],
            y=daily_metrics['Discount_Amount'],
            mode='lines+markers',
            name='Daily Discounts',
            line=dict(color='#FF6B6B', width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Discounts:</b> $%{y:,.2f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Discount Amount ($)", secondary_y=True)
    
    fig.update_layout(
        title_text='Daily Revenue vs Promotions Impact',
        title_font_size=20,
        title_x=0.5,
        title_y=0.95,
        hovermode='x unified',
        margin=dict(t=80, b=50, l=80, r=50)
    )
    
    return fig

def main():
    """Main Clarus coffee shop dashboard function"""
    # Header
    st.markdown('<h1 class="main-header"> Clarus Coffee Analytics</h1>', unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div style="background-color: #FFF8DC; padding: 1rem; border-radius: 10px; border-left: 5px solid #8B4513; margin-bottom: 2rem;">
        <h4>Welcome to Clarus Coffee Shop Analytics!</h4>
        <p>Transform your coffee shop data into actionable business insights with our comprehensive analytics platform.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Clean data options
    st.sidebar.header("Data Source")
    
    # Simple two-option approach
    data_option = st.sidebar.radio(
        "Choose your data:",
        ["View Sample Data", "Upload Your Data"],
        help="Select how you'd like to explore the Clarus Coffee Analytics platform"
    )
    
    if data_option == "Upload Your Data":
        st.sidebar.markdown("**Upload your POS data:**")
        uploaded_file = st.sidebar.file_uploader(
            "Choose CSV or Excel file", 
            type=['csv', 'xlsx'],
            help="Supports CSV and Excel (.xlsx) files"
        )
        st.sidebar.markdown("*Your data stays secure and private.*")
    else:
        uploaded_file = None
    
    # Load data based on user choice
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.sidebar.success("Your data loaded successfully!")
        st.sidebar.info(f"Analyzing {len(df):,} transactions")
    else:
        # Try to load sample data
        try:
            df = load_data('synthetic_POS_data_coffee.xlsx')
            if data_option == "View Sample Data":
                st.sidebar.success("Sample data loaded")
        except:
            try:
                df = load_data('synthetic_POS_data_coffee.csv')
                if data_option == "View Sample Data":
                    st.sidebar.success("Sample data loaded")
            except:
                st.error("No data available. Please upload your coffee shop POS data to get started.")
                st.info("""
                **Required columns:** Transaction_ID, Timestamp, Employee_ID, Customer_ID, Item_Name, Category, Quantity, Unit_Price, Discount, Payment_Method, Total_Amount, Tax, Tip
                """)
                return
    
    if df is None:
        return
    
    # Display data info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Overview")
    st.sidebar.write(f"**Total Transactions:** {len(df):,}")
    st.sidebar.write(f"**Date Range:** {df['Date'].min()} to {df['Date'].max()}")
    st.sidebar.write(f"**Total Revenue:** ${df['Revenue'].sum():,.2f}")
    
    # Sidebar filters
    st.sidebar.header("Filter Your Data")
    
    # Month filter
    months_available = sorted(df['Month'].unique())
    month_names = [str(month) for month in months_available]
    
    selected_month_idx = st.sidebar.selectbox(
        "Select Month",
        range(len(months_available)),
        format_func=lambda x: month_names[x],
        index=len(months_available)-1  # Default to latest month
    )
    selected_month = months_available[selected_month_idx]
    
    # Filter data by selected month
    filtered_df = df[df['Month'] == selected_month]
    
    # Date range filter within the month
    min_date = filtered_df['Date'].min()
    max_date = filtered_df['Date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['Date'] >= date_range[0]) & 
            (filtered_df['Date'] <= date_range[1])
        ]
    
    # Key Metrics Dashboard
    st.header("Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = filtered_df['Revenue'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col2:
        total_transactions = filtered_df['Transaction_ID'].nunique()
        st.metric("Total Orders", f"{total_transactions:,}")
    
    with col3:
        avg_transaction = filtered_df['Revenue'].mean()
        st.metric("Avg Order Value", f"${avg_transaction:.2f}")
    
    with col4:
        unique_customers = filtered_df['Customer_ID'].nunique()
        st.metric("Unique Customers", f"{unique_customers:,}")
    
    # Coffee shop specific metrics
    create_coffee_specific_metrics(filtered_df)
    
    # Main Charts Section
    st.header("Sales Analytics")
    
    # Daily Sales Trend
    st.plotly_chart(create_daily_sales_trend(filtered_df), use_container_width=True)
    
    # Hourly Heatmap
    st.plotly_chart(create_hourly_heatmap(filtered_df), use_container_width=True)
    
    # Category Performance
    st.subheader("Product Category Performance")
    create_category_breakdown(filtered_df)
    
    # Two column layout for additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_top_products(filtered_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_payment_method_chart(filtered_df), use_container_width=True)
    
    # Employee Performance
    st.plotly_chart(create_employee_performance(filtered_df), use_container_width=True)
    
    # Customer and Business Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_customer_frequency(filtered_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_discount_impact(filtered_df), use_container_width=True)
    
    # Raw Data Section
    with st.expander("View Raw Transaction Data"):
        st.subheader("Recent Transactions")
        display_df = filtered_df.sort_values('Timestamp', ascending=False).head(100)
        
        # Use st.table instead of st.dataframe (doesn't require PyArrow)
        st.table(
            display_df[['Timestamp', 'Item_Name', 'Category', 'Quantity', 'Unit_Price', 'Total_Amount', 'Payment_Method']].head(20)
        )
        
        if len(display_df) > 20:
            st.info(f"Showing first 20 of {len(display_df)} transactions. Download CSV below for complete data.")
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Clarus Analytics Report (CSV)",
            data=csv,
            file_name=f"clarus_coffee_analytics_{selected_month}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="clarus-footer">
        <h4>Clarus Analytics Platform</h4>
        <p><em>Built with Streamlit & Plotly | © 2024 Clarus Business Intelligence</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()