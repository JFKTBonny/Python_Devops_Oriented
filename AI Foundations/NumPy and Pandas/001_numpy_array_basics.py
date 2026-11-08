import numpy as np


ticket_lengths = np.array([18, 42, 31, 76, 25, 54], dtype=np.float64)

print(f"Average length: {ticket_lengths.mean():.1f} characters")
print(f"Median length: {np.median(ticket_lengths):.1f} characters")
print(f"Standard deviation: {ticket_lengths.std():.1f}")

long_tickets = ticket_lengths[ticket_lengths >= 40]
print(f"Tickets with at least 40 characters: {long_tickets.tolist()}")
