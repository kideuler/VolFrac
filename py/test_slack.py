from Model import *

# Example usage
if __name__ == "__main__":
    # Message text
    message = "Training completed! Here are the results:"
    
    # Path to the file you want to upload
    file_path = "models/final_loss_plot.png"
    
    # Optional: custom title and comment for the file
    file_title = "Training Loss Plot"
    file_comment = "This plot shows the training and validation loss over epochs."
    
    # Send the message with file
    send_message_with_file_to_slack(message, file_path, file_title, file_comment)