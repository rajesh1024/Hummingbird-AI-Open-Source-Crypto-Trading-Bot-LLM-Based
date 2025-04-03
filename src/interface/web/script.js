class SignalStreamer {
    constructor() {
        this.eventSource = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000; // Start with 2 seconds
        
        // Check for required DOM elements
        this.checkDOMElements();
        
        this.setupEventListeners();
    }

    checkDOMElements() {
        // Check for all required elements
        const requiredElements = {
            'startBtn': 'Start Button',
            'stopBtn': 'Stop Button',
            'symbolSelect': 'Symbol Select',
            'modeSelect': 'Mode Select',
            'status': 'Status Indicator',
            'terminal': 'Terminal Output',
            'timestamp': 'Timestamp Display',
            'signalType': 'Signal Type Display',
            'confidence': 'Confidence Display',
            'entryPrice': 'Entry Price Display',
            'stopLoss': 'Stop Loss Display',
            'takeProfit': 'Take Profit Display',
            'reasoning': 'Reasoning Display',
            'activePositions': 'Active Positions Display',
            'tradingStats': 'Trading Statistics Display'
        };

        console.log('Checking for required DOM elements...');
        const missingElements = [];

        for (const [id, description] of Object.entries(requiredElements)) {
            const element = document.getElementById(id);
            if (!element) {
                missingElements.push(`${description} (${id})`);
                console.warn(`Missing required element: ${description} (${id})`);
            } else {
                console.log(`Found element: ${description} (${id})`);
            }
        }

        if (missingElements.length > 0) {
            console.error('Missing required elements:', missingElements);
            this.appendToTerminal(`Warning: Missing required elements: ${missingElements.join(', ')}`);
        }
    }

    setupEventListeners() {
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');

        if (startBtn) {
            startBtn.addEventListener('click', () => this.startStreaming());
            console.log('Start button event listener added');
        } else {
            console.error('Start button not found');
        }

        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopStreaming());
            console.log('Stop button event listener added');
        } else {
            console.error('Stop button not found');
        }
    }

    async startStreaming() {
        if (this.isConnected) return;

        const symbolSelect = document.getElementById('symbolSelect');
        const modeSelect = document.getElementById('modeSelect');

        if (!symbolSelect) {
            console.error('Symbol select not found');
            return;
        }

        if (!modeSelect) {
            console.error('Mode select not found');
            return;
        }

        const symbol = symbolSelect.value;
        const mode = modeSelect.value || 'scalping';
        const encodedSymbol = encodeURIComponent(symbol).replace(/%2F/g, '_');

        try {
            const sseUrl = `http://localhost:8000/stream/${encodedSymbol}?mode=${mode}`;
            
            console.log('Attempting to connect to SSE:', sseUrl);
            this.appendToTerminal(`Connecting to SSE at ${sseUrl}...`);
            
            // First, check if the server is running
            try {
                console.log('Checking server availability...');
                const response = await fetch(sseUrl, {
                    method: 'GET',
                    headers: {
                        'Accept': 'text/event-stream',
                        'Cache-Control': 'no-cache',
                    },
                });
                
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Server returned ${response.status}: ${response.statusText}\nDetails: ${errorText}`);
                }
                
                console.log('Server check successful');
            } catch (error) {
                console.error('Server check failed:', error);
                this.appendToTerminal(`Server check failed: ${error.message}`);
                this.appendToTerminal('Please ensure the FastAPI server is running on port 8000');
                this.handleConnectionError();
                return;
            }
            
            // Create EventSource with proper configuration
            console.log('Creating EventSource...');
            this.eventSource = new EventSource(sseUrl);
            
            // Set up handlers before checking readyState
            this.setupEventSourceHandlers();
            
            // Check initial connection state
            if (this.eventSource.readyState === EventSource.CLOSED) {
                throw new Error('Failed to establish initial connection');
            }
            
            this.reconnectAttempts = 0;
            this.reconnectDelay = 2000; // Reset delay
        } catch (error) {
            console.error('Error creating EventSource:', error);
            this.appendToTerminal(`Error: Failed to create SSE connection - ${error.message}`);
            this.handleConnectionError();
        }
    }

    setupEventSourceHandlers() {
        this.eventSource.onopen = () => {
            console.log('SSE connection opened');
            this.isConnected = true;
            this.updateStatus('Connected');
            this.appendToTerminal('Connected to signal stream...');
            this.reconnectAttempts = 0;
            this.reconnectDelay = 2000; // Reset delay
        };

        // Handle terminal output events
        this.eventSource.addEventListener('terminal', (event) => {
            try {
                console.log('Received terminal event:', event.data);
                const data = JSON.parse(event.data);
                if (data.message) {
                    this.appendToTerminal(data.message);
                    console.log('Terminal message:', data.message);
                    
                    // Check for signal data in terminal messages
                    if (data.message.includes('Signal Analysis')) {
                        console.log('Found Signal Analysis in terminal message');
                        const signalData = this.parseSignalAnalysis(data.message);
                        console.log('Parsed signal data from terminal:', signalData);
                        if (signalData) {
                            this.updateSignalDisplay(signalData);
                        } else {
                            console.warn('Failed to parse signal data from terminal message');
                        }
                    }
                }
            } catch (error) {
                console.error('Error parsing terminal message:', error);
            }
        });

        // Handle signal data events
        this.eventSource.addEventListener('signal', (event) => {
            try {
                console.log('Received signal event:', event.data);
                const data = JSON.parse(event.data);
                console.log('Parsed signal data:', data);
                
                // Format the data for display
                const formattedData = {
                    signal_type: data.signal || 'N/A',
                    confidence: data.confidence || 0,
                    entry_price: data.entry_price || 0,
                    stop_loss: data.stop_loss || 0,
                    take_profit: data.take_profit || 0,
                    reasoning: data.reasoning || 'N/A',
                    current_price: data.current_price || 0,
                    active_positions: data.active_positions || 0,
                    position_management: data.position_management || {}
                };
                
                console.log('Formatted signal data:', formattedData);
                this.updateSignalDisplay(formattedData);
                
                // Update active positions if available
                if (formattedData.active_positions !== undefined) {
                    const positionsElement = document.getElementById('activePositions');
                    if (positionsElement) {
                        positionsElement.textContent = `Active Positions: ${formattedData.active_positions}`;
                    }
                }

                // Update position management if available
                if (formattedData.position_management && Object.keys(formattedData.position_management).length > 0) {
                    const positionAction = formattedData.position_management.action || 'N/A';
                    const stopLossAdjustment = formattedData.position_management.stop_loss_adjustment || 'N/A';
                    const takeProfitAdjustment = formattedData.position_management.take_profit_adjustment || 'N/A';
                    const riskRewardRatio = formattedData.position_management.risk_reward_ratio || 'N/A';

                    // Log position management details
                    console.log('Position Management:', {
                        action: positionAction,
                        stopLossAdjustment,
                        takeProfitAdjustment,
                        riskRewardRatio
                    });
                }
            } catch (error) {
                console.error('Error parsing signal data:', error);
            }
        });

        // Handle default messages (like connection status)
        this.eventSource.onmessage = (event) => {
            try {
                console.log('Received default message:', event.data);
                const data = JSON.parse(event.data);
                if (data.status === 'connected') {
                    this.appendToTerminal(data.message);
                }
            } catch (error) {
                console.error('Error parsing message:', error);
            }
        };

        this.eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            console.log('EventSource readyState:', this.eventSource.readyState);
            this.handleConnectionError();
        };
    }

    handleConnectionError() {
        this.isConnected = false;
        this.updateStatus('Disconnected');
        this.appendToTerminal('Error: Connection failed');
        
        // Close the existing connection if it exists
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        
        // Attempt to reconnect if not manually stopped
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            this.appendToTerminal(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            
            // Exponential backoff
            setTimeout(() => this.startStreaming(), this.reconnectDelay);
            this.reconnectDelay *= 2; // Double the delay for next attempt
        } else {
            this.appendToTerminal('Max reconnection attempts reached. Please try again manually.');
            this.updateStatus('Failed');
        }
    }

    stopStreaming() {
        if (this.eventSource) {
            this.eventSource.close();
            this.isConnected = false;
            this.updateStatus('Disconnected');
            this.appendToTerminal('Streaming stopped');
            this.eventSource = null;
            this.reconnectAttempts = 0;
            this.reconnectDelay = 2000;
        }
    }

    updateStatus(status) {
        const statusElement = document.getElementById('status');
        statusElement.textContent = status;
        statusElement.className = `status-indicator ${status.toLowerCase()}`;
    }

    parseSignalMessage(message) {
        try {
            // Example message format:
            // Signal: LONG | Entry: 50000 | SL: 49000 | TP: 52000 | Confidence: 0.85 | Reasoning: Strong uptrend
            const signalMatch = message.match(/Signal: (.*?) \|/);
            const entryMatch = message.match(/Entry: (.*?) \|/);
            const slMatch = message.match(/SL: (.*?) \|/);
            const tpMatch = message.match(/TP: (.*?) \|/);
            const confidenceMatch = message.match(/Confidence: (.*?) \|/);
            const reasoningMatch = message.match(/Reasoning: (.*)/);

            if (signalMatch) {
                return {
                    signal_type: signalMatch[1].trim(),
                    entry_price: parseFloat(entryMatch?.[1] || 0),
                    stop_loss: parseFloat(slMatch?.[1] || 0),
                    take_profit: parseFloat(tpMatch?.[1] || 0),
                    confidence: parseFloat(confidenceMatch?.[1] || 0),
                    reasoning: reasoningMatch?.[1]?.trim() || ''
                };
            }
        } catch (error) {
            console.error('Error parsing signal message:', error);
        }
        return null;
    }

    parseSignalAnalysis(message) {
        try {
            console.log('Parsing signal analysis from message:', message);
            
            // Extract data from the analysis table
            const currentPriceMatch = message.match(/Current Price\s+\|\s+([\d.]+)/);
            const signalTypeMatch = message.match(/Signal Type\s+\|\s+(\w+)/);
            const confidenceMatch = message.match(/Confidence\s+\|\s+([\d.]+)/);
            const entryPriceMatch = message.match(/Entry Price\s+\|\s+([\d.]+)/);
            const stopLossMatch = message.match(/Stop Loss\s+\|\s+([\d.]+)/);
            const takeProfitMatch = message.match(/Take Profit\s+\|\s+([\d.]+)/);
            const reasoningMatch = message.match(/Reasoning\s+\|\s+([^|]+)/);

            console.log('Regex matches:', {
                currentPrice: currentPriceMatch?.[1],
                signalType: signalTypeMatch?.[1],
                confidence: confidenceMatch?.[1],
                entryPrice: entryPriceMatch?.[1],
                stopLoss: stopLossMatch?.[1],
                takeProfit: takeProfitMatch?.[1],
                reasoning: reasoningMatch?.[1]
            });

            if (signalTypeMatch) {
                const signalData = {
                    signal_type: signalTypeMatch[1],
                    entry_price: parseFloat(entryPriceMatch?.[1] || 0),
                    stop_loss: parseFloat(stopLossMatch?.[1] || 0),
                    take_profit: parseFloat(takeProfitMatch?.[1] || 0),
                    confidence: parseFloat(confidenceMatch?.[1] || 0),
                    reasoning: reasoningMatch?.[1]?.trim() || ''
                };
                console.log('Successfully parsed signal data:', signalData);
                return signalData;
            } else {
                console.warn('No signal type found in message');
            }
        } catch (error) {
            console.error('Error parsing signal analysis:', error);
            console.error('Error details:', error.stack);
        }
        return null;
    }

    updateSignalDisplay(data) {
        try {
            console.log('Starting updateSignalDisplay with data:', data);
            
            // Get all required elements
            const elements = {
                timestamp: document.getElementById('timestamp'),
                signalType: document.getElementById('signalType'),
                confidence: document.getElementById('confidence'),
                entryPrice: document.getElementById('entryPrice'),
                stopLoss: document.getElementById('stopLoss'),
                takeProfit: document.getElementById('takeProfit'),
                reasoning: document.getElementById('reasoning')
            };

            // Log which elements were found
            console.log('Found HTML elements:', {
                timestamp: !!elements.timestamp,
                signalType: !!elements.signalType,
                confidence: !!elements.confidence,
                entryPrice: !!elements.entryPrice,
                stopLoss: !!elements.stopLoss,
                takeProfit: !!elements.takeProfit,
                reasoning: !!elements.reasoning
            });

            // Update timestamp
            if (elements.timestamp) {
                elements.timestamp.textContent = new Date().toLocaleString();
                console.log('Updated timestamp:', elements.timestamp.textContent);
            } else {
                console.warn('Timestamp element not found');
            }

            // Update signal type with color coding
            if (elements.signalType) {
                const signalValue = data.signal_type || data.signal || '-';
                elements.signalType.textContent = signalValue;
                elements.signalType.className = `signal-type ${signalValue.toLowerCase()}`;
                console.log('Updated signal type:', signalValue);
            } else {
                console.warn('Signal type element not found');
            }

            // Update confidence as percentage
            if (elements.confidence) {
                const confidenceValue = data.confidence ? `${(data.confidence * 100).toFixed(2)}%` : '-';
                elements.confidence.textContent = confidenceValue;
                console.log('Updated confidence:', confidenceValue);
            } else {
                console.warn('Confidence element not found');
            }

            // Update prices with proper formatting
            if (elements.entryPrice) {
                const entryValue = data.entry_price ? `$${data.entry_price.toFixed(2)}` : '-';
                elements.entryPrice.textContent = entryValue;
                console.log('Updated entry price:', entryValue);
            } else {
                console.warn('Entry price element not found');
            }

            if (elements.stopLoss) {
                const slValue = data.stop_loss ? `$${data.stop_loss.toFixed(2)}` : '-';
                elements.stopLoss.textContent = slValue;
                console.log('Updated stop loss:', slValue);
            } else {
                console.warn('Stop loss element not found');
            }

            if (elements.takeProfit) {
                const tpValue = data.take_profit ? `$${data.take_profit.toFixed(2)}` : '-';
                elements.takeProfit.textContent = tpValue;
                console.log('Updated take profit:', tpValue);
            } else {
                console.warn('Take profit element not found');
            }

            // Update reasoning
            if (elements.reasoning) {
                const reasoningValue = data.reasoning || '-';
                elements.reasoning.textContent = reasoningValue;
                console.log('Updated reasoning:', reasoningValue);
            } else {
                console.warn('Reasoning element not found');
            }

            // Add visual feedback for the signal section
            const signalDetails = document.querySelector('.signal-details');
            if (signalDetails) {
                signalDetails.classList.add('has-signal');
                console.log('Added visual feedback to signal details');
                // Remove the class after 5 seconds to indicate signal age
                setTimeout(() => {
                    signalDetails.classList.remove('has-signal');
                    console.log('Removed visual feedback from signal details');
                }, 5000);
            } else {
                console.warn('Signal details element not found');
            }

            // Log the complete update for debugging
            console.log('Signal display update complete:', {
                timestamp: elements.timestamp?.textContent,
                signalType: elements.signalType?.textContent,
                confidence: elements.confidence?.textContent,
                entryPrice: elements.entryPrice?.textContent,
                stopLoss: elements.stopLoss?.textContent,
                takeProfit: elements.takeProfit?.textContent,
                reasoning: elements.reasoning?.textContent
            });
        } catch (error) {
            console.error('Error updating signal display:', error);
            console.error('Error details:', error.stack);
        }
    }

    updateActivePositions(message) {
        const positionsContainer = document.getElementById('activePositions');
        if (!positionsContainer) return;

        try {
            // Extract positions from the message
            const positionsMatch = message.match(/Active Positions: (.*)/);
            if (positionsMatch) {
                const positions = positionsMatch[1];
                positionsContainer.textContent = positions || 'No active positions';
                positionsContainer.className = positions ? 'has-positions' : 'no-positions';
            }
        } catch (error) {
            console.error('Error updating active positions:', error);
            positionsContainer.textContent = 'Error loading positions';
        }
    }

    updateTradingStatistics(message) {
        const statsContainer = document.getElementById('tradingStats');
        if (!statsContainer) return;

        try {
            // Extract statistics from the message
            const statsMatch = message.match(/Trading Statistics: (.*)/);
            if (statsMatch) {
                const stats = statsMatch[1];
                statsContainer.textContent = stats || 'No trading statistics available';
                statsContainer.className = stats ? 'has-stats' : 'no-stats';
            }
        } catch (error) {
            console.error('Error updating trading statistics:', error);
            statsContainer.textContent = 'Error loading statistics';
        }
    }

    appendToTerminal(message) {
        const terminal = document.getElementById('terminal');
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        
        // Add different classes based on message type
        if (message.startsWith('INFO:')) {
            logEntry.classList.add('info-message');
        } else if (message.startsWith('Debug:')) {
            logEntry.classList.add('debug-message');
        } else if (message.startsWith('WARNING:')) {
            logEntry.classList.add('warning-message');
        } else if (message.startsWith('ERROR:')) {
            logEntry.classList.add('error-message');
        }
        
        logEntry.textContent = `[${timestamp}] ${message}`;
        terminal.appendChild(logEntry);
        terminal.scrollTop = terminal.scrollHeight;
    }
}

// Initialize the signal streamer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.signalStreamer = new SignalStreamer();
}); 