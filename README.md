## Language Analysis Module
This module processes LENA ITS files and produces language developement metrics and visualizations over a 24 hour period

The module performs the following:
- Parse a LENA .its XML file
- Extract segments and coversations
- Compute the following language metrics:
  - Adult Word Count (AWC)
  - Child Vocalization Count (CVC)
  - Conversational Turn Count (CTC)
- Save extracted data to CSV files
- Compute daily totals for each aforementioned metric
- Generate visualizations in the form of bar graphs over the 24 hour period

## Folder Structure
Language Analysis/
--> src/
    --> cli/ 
        --> run_extraction.py
        --> run_daily_summary.py
    --> core/
        --> its_loader.py
        --> its_extraction.py
        --> daily_summary.py
        --> statistics.py
    --> plots/
        --> daily_summary_plot.py
    --> __init__.py
    
## Running the Language Analysis Pipeline
# Navigate to the src directory
Run:
- cd "Language Analysis/src"

# Step 1: Extract ITS Data
Run:
- python -m cli.run_extraction.py --its "its_file_path"

This will:
- Parse the ITS XML file
- Extract segments and conversations
- Compute summary statistics
- Save extracted data to CSV files

# Step 2 — Generate Daily Summary
Run:
- python cli/run_daily_summary.py

This will:
- Load extracted CSV files
- Calculate daily AWC, CVC, and CTC
- Generate a visualization of daily totals

## Outputs
Results can be found in the following directories:
- `lena_extraction_out/` — CSV file outputs
- `lena_visualizations/` — full plot output

        