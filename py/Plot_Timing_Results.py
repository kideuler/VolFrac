import matplotlib.pyplot as plt
import numpy as np
import re
import os

def extract_data_from_tex(filepath):
    """Extract data from a LaTeX table file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract the header (column names)
    header_match = re.search(r'Sizes & (.*) \\\\', content)
    if header_match:
        headers = ["Sizes"] + header_match.group(1).split(' & ')
    else:
        print("Couldn't extract headers from table")
        return None, None
    
    # Extract the rows
    rows = []
    for line in content.split('\n'):
        if line.startswith('\\hline'):
            continue
        if '&' in line:
            row = line.strip().replace('\\\\', '').replace('\\hline', '').split('&')
            row = [x.strip() for x in row]
            if row[0].isdigit() or row[0].replace('.', '', 1).isdigit():
                rows.append(row)
    
    return headers, rows

def create_loglog_plot(headers, data, output_file=None):
    """Create a log-log plot from the extracted data."""
    plt.figure(figsize=(10, 8))
    
    sizes = np.array([float(row[0]) for row in data])
    sizes_squared = sizes  # N^2
    
    markers = ['o', 's', '^', 'D', '*', 'x', '+', 'v']
    colors = plt.cm.Dark2(np.linspace(0, 1, len(headers)-1))
    
    for i, method in enumerate(headers[1:]):
        times = np.array([float(row[i+1]) for row in data])
        plt.loglog(sizes_squared, times, marker=markers[i % len(markers)], 
                   label=method, color=colors[i], linewidth=2, markersize=8)
    
    # Add ideal convergence rate reference lines
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('$\sqrt{N}$', fontsize=18)
    plt.ylabel('Time (Seconds)', fontsize=18)
    # plt.title('Accuracy Comparison for Different Volume Fraction Methods', fontsize=16)
    plt.legend(fontsize=12)
    
    # Add a text annotation for convergence rates
    # plt.figtext(0.15, 0.15, "Steeper slope = faster convergence", bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.draw()
    # Make sure it gets displayed by flushing events
    plt.pause(0.5)

def main(tex_file):
    if not os.path.exists(tex_file):
        print(f"File not found: {tex_file}")
        return
    
    # Extract data from the TeX file
    headers, data = extract_data_from_tex(tex_file)
    if headers is None or data is None:
        print("Failed to extract data from the TeX file.")
        return
    
    # Create the log-log plot
    output_file = tex_file.replace('.tex', '.png')
    create_loglog_plot(headers, data, output_file)
    print(f"Plot saved to {output_file}")
    
    # Calculate and print convergence rates
    print("\nApproximate convergence rates:")
    sizes = np.array([float(row[0]) for row in data])
    for i, method in enumerate(headers[1:]):
        errors = np.array([float(row[i+1]) for row in data])
        # Use linear regression on log-log scale to estimate convergence rate
        if len(sizes) >= 3:  # Need at least 3 points for meaningful regression
            log_sizes = np.log(sizes)
            log_errors = np.log(errors)
            # Remove any -inf values from log of zeros
            valid = np.isfinite(log_errors)
            if np.sum(valid) >= 3:
                p = np.polyfit(log_sizes[valid], log_errors[valid], 1)
                rate = -p[0]  # Negative since we expect error to decrease with size
                print(f"{method}: {rate:.2f}")
            else:
                print(f"{method}: Not enough valid data points")

if __name__ == "__main__":
    main(os.path.join('build', 'results', 'Timing_table_500.tex'))
    main(os.path.join('build', 'results', 'Timing_table_5000.tex'))
    main(os.path.join('build', 'results', 'Timing_table_50000.tex'))