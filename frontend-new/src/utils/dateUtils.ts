export const convertUTCToIST = (utcDateString: string): { formatted: string; timestamp: string } => {
  // Create a date object from the UTC string
  const utcDate = new Date(utcDateString);
  
  // Add 5 hours and 30 minutes for IST conversion
  const istDate = new Date(utcDate.getTime() + (5.5 * 60 * 60 * 1000));
  
  // Format the date
  const day = istDate.getDate().toString().padStart(2, '0');
  const month = (istDate.getMonth() + 1).toString().padStart(2, '0');
  const year = istDate.getFullYear();
  const hours = istDate.getHours().toString().padStart(2, '0');
  const minutes = istDate.getMinutes().toString().padStart(2, '0');
  const seconds = istDate.getSeconds().toString().padStart(2, '0');

  return {
    formatted: `${day}/${month}/${year}, ${hours}:${minutes}:${seconds}`,
    timestamp: istDate.toISOString()
  };
};

export const formatDuration = (createdAt: string | null): number | null => {
  if (!createdAt) return null;
  const created = new Date(createdAt);
  const now = new Date();
  const diffInMinutes = Math.floor((now.getTime() - created.getTime()) / (1000 * 60));
  return diffInMinutes;
}; 