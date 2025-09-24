// Modern Heart Attack Risk Prediction App
class HeartRiskApp {
    constructor() {
        this.currentStep = 1;
        this.totalSteps = 4;
        this.API_BASE_URL = 'http://localhost:5000';
        
        // DOM Elements
        this.form = null;
        this.prevBtn = null;
        this.nextBtn = null;
        this.predictBtn = null;
        this.progressFill = null;
        this.resultsContainer = null;
        this.errorContainer = null;
        
        // Form validation
        this.validationRules = {
            Age: { min: 18, max: 100, required: true },
            Cholesterol: { min: 100, max: 400, required: true },
            'Heart Rate': { min: 40, max: 200, required: true },
            'Exercise Hours Per Week': { min: 0, max: 20, required: true },
            'Blood Pressure': { pattern: /^\d{2,3}\/\d{2,3}$/, required: true }
        };
        
        this.init();
    }
    
    init() {
        document.addEventListener('DOMContentLoaded', () => {
            this.initializeElements();
            this.attachEventListeners();
            this.updateProgress();
            this.testAPIConnection();
            
            console.log('🚀 Modern Heart Risk App initialized');
        });
    }
    
    initializeElements() {
        this.form = document.getElementById('prediction-form');
        this.prevBtn = document.getElementById('prev-btn');
        this.nextBtn = document.getElementById('next-btn');
        this.predictBtn = document.getElementById('predict-btn');
        this.progressFill = document.getElementById('progress-fill');
        this.resultsContainer = document.getElementById('results-container');
        this.errorContainer = document.getElementById('error-container');
        
        // Initialize stress slider
        this.initializeStressSlider();
    }
    
    attachEventListeners() {
        // Form navigation
        this.prevBtn?.addEventListener('click', () => this.previousStep());
        this.nextBtn?.addEventListener('click', () => this.nextStep());
        this.form?.addEventListener('submit', (e) => this.handleFormSubmit(e));
        
        // Real-time validation
        this.attachValidationListeners();
        
        // Stress slider
        const stressSlider = document.getElementById('Stress Level');
        stressSlider?.addEventListener('input', (e) => this.updateStressValue(e.target.value));
        
        // Radio button animations
        this.attachRadioListeners();
    }
    
    attachValidationListeners() {
        // Add validation listeners to all inputs
        const inputs = this.form?.querySelectorAll('input, select');
        inputs?.forEach(input => {
            input.addEventListener('blur', () => this.validateField(input));
            input.addEventListener('input', () => this.clearFieldError(input));
        });
    }
    
    attachRadioListeners() {
        const radioInputs = document.querySelectorAll('input[type="radio"]');
        radioInputs.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.animateRadioSelection(e.target);
            });
        });
    }
    
    initializeStressSlider() {
        const stressSlider = document.getElementById('Stress Level');
        const stressValue = document.getElementById('stress-value');
        
        if (stressSlider && stressValue) {
            stressValue.textContent = stressSlider.value;
            this.updateStressSliderColor(stressSlider.value);
        }
    }
    
    updateStressValue(value) {
        const stressValue = document.getElementById('stress-value');
        if (stressValue) {
            stressValue.textContent = value;
            this.updateStressSliderColor(value);
        }
    }
    
    updateStressSliderColor(value) {
        const slider = document.getElementById('Stress Level');
        if (!slider) return;
        
        const percentage = ((value - 1) / 9) * 100;
        const hue = 120 - (value - 1) * 13.33; // Green to red
        
        slider.style.background = `linear-gradient(90deg, 
            hsl(${hue}, 70%, 50%) 0%, 
            hsl(${hue}, 70%, 50%) ${percentage}%, 
            rgba(255, 255, 255, 0.2) ${percentage}%, 
            rgba(255, 255, 255, 0.2) 100%)`;
    }
    
    animateRadioSelection(radio) {
        const radioGroup = radio.closest('.radio-group');
        if (!radioGroup) return;
        
        // Remove animation from all labels in the group
        radioGroup.querySelectorAll('.radio-label').forEach(label => {
            label.style.transform = '';
        });
        
        // Animate selected label
        const selectedLabel = radio.nextElementSibling;
        if (selectedLabel) {
            selectedLabel.style.transform = 'scale(1.05)';
            setTimeout(() => {
                selectedLabel.style.transform = '';
            }, 200);
        }
    }
    
    validateField(field) {
        const fieldName = field.name || field.id;
        const value = field.value.trim();
        const rules = this.validationRules[fieldName];
        
        if (!rules) return true;
        
        let isValid = true;
        let errorMessage = '';
        
        // Required validation
        if (rules.required && !value) {
            isValid = false;
            errorMessage = 'This field is required';
        }
        
        // Pattern validation
        else if (rules.pattern && value && !rules.pattern.test(value)) {
            isValid = false;
            errorMessage = 'Please use the correct format';
        }
        
        // Range validation
        else if ((rules.min || rules.max) && value) {
            const numValue = parseFloat(value);
            if (rules.min && numValue < rules.min) {
                isValid = false;
                errorMessage = `Minimum value is ${rules.min}`;
            } else if (rules.max && numValue > rules.max) {
                isValid = false;
                errorMessage = `Maximum value is ${rules.max}`;
            }
        }
        
        this.showFieldFeedback(field, isValid, errorMessage);
        return isValid;
    }
    
    showFieldFeedback(field, isValid, message) {
        const feedback = field.parentElement.querySelector('.input-feedback');
        if (!feedback) return;
        
        feedback.textContent = message;
        feedback.className = `input-feedback ${isValid ? 'success' : 'error'}`;
        
        // Add visual feedback to field
        field.style.borderColor = isValid ? 
            'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)';
    }
    
    clearFieldError(field) {
        const feedback = field.parentElement.querySelector('.input-feedback');
        if (feedback) {
            feedback.textContent = '';
            feedback.className = 'input-feedback';
        }
        field.style.borderColor = '';
    }
    
    validateCurrentStep() {
        const currentStepElement = document.querySelector(`.form-step[data-step="${this.currentStep}"]`);
        if (!currentStepElement) return false;
        
        const fields = currentStepElement.querySelectorAll('input, select');
        let isValid = true;
        
        fields.forEach(field => {
            if (field.type === 'radio') {
                const radioGroup = currentStepElement.querySelectorAll(`input[name="${field.name}"]`);
                const isRadioGroupValid = Array.from(radioGroup).some(radio => radio.checked);
                if (!isRadioGroupValid) {
                    isValid = false;
                    this.showFieldFeedback(field, false, 'Please select an option');
                }
            } else {
                if (!this.validateField(field)) {
                    isValid = false;
                }
            }
        });
        
        return isValid;
    }
    
    nextStep() {
        if (!this.validateCurrentStep()) {
            this.showNotification('Please fill in all required fields correctly', 'error');
            return;
        }
        
        if (this.currentStep < this.totalSteps) {
            this.currentStep++;
            this.updateStepDisplay();
            this.updateProgress();
            this.updateNavigationButtons();
        }
    }
    
    previousStep() {
        if (this.currentStep > 1) {
            this.currentStep--;
            this.updateStepDisplay();
            this.updateProgress();
            this.updateNavigationButtons();
        }
    }
    
    updateStepDisplay() {
        // Hide all steps
        document.querySelectorAll('.form-step').forEach(step => {
            step.classList.remove('active');
        });
        
        // Show current step
        const currentStepElement = document.querySelector(`.form-step[data-step="${this.currentStep}"]`);
        currentStepElement?.classList.add('active');
        
        // Update progress steps
        document.querySelectorAll('.step').forEach((step, index) => {
            step.classList.toggle('active', index + 1 === this.currentStep);
        });
    }
    
    updateProgress() {
        const progress = (this.currentStep / this.totalSteps) * 100;
        if (this.progressFill) {
            this.progressFill.style.width = `${progress}%`;
        }
    }
    
    updateNavigationButtons() {
        if (this.prevBtn) {
            this.prevBtn.disabled = this.currentStep === 1;
        }
        
        if (this.nextBtn && this.predictBtn) {
            if (this.currentStep === this.totalSteps) {
                this.nextBtn.style.display = 'none';
                this.predictBtn.style.display = 'flex';
            } else {
                this.nextBtn.style.display = 'flex';
                this.predictBtn.style.display = 'none';
            }
        }
    }
    
    async handleFormSubmit(event) {
        event.preventDefault();
        
        if (!this.validateCurrentStep()) {
            this.showNotification('Please fill in all required fields correctly', 'error');
            return;
        }
        
        this.hideResults();
        this.hideError();
        
        const formData = this.collectFormData();
        if (!formData) {
            this.showError('Please fill in all required fields correctly.');
            return;
        }
        
        this.setLoadingState(true);
        
        try {
            await this.makePrediction(formData);
        } catch (error) {
            console.error('Prediction failed:', error);
            this.showError(error.message);
        } finally {
            this.setLoadingState(false);
        }
    }
    
    collectFormData() {
        try {
            const formData = new FormData(this.form);
            const data = {};
            
            // Handle regular form fields
            for (let [key, value] of formData.entries()) {
                data[key] = value;
            }
            
            // Validate required fields
            const requiredFields = [
                'Age', 'Gender', 'Cholesterol', 'Blood Pressure', 'Heart Rate',
                'Smoking', 'Exercise Hours Per Week', 'Stress Level',
                'Obesity', 'Diabetes', 'Previous Heart Problems', 'Medication Use'
            ];
            
            for (let field of requiredFields) {
                if (!data[field] || data[field] === '') {
                    console.error(`Missing field: ${field}`);
                    return null;
                }
            }
            
            // Convert numeric fields
            data['Age'] = parseInt(data['Age']);
            data['Cholesterol'] = parseInt(data['Cholesterol']);
            data['Heart Rate'] = parseInt(data['Heart Rate']);
            data['Exercise Hours Per Week'] = parseFloat(data['Exercise Hours Per Week']);
            data['Stress Level'] = parseInt(data['Stress Level']);
            
            console.log('✅ Form data collected:', data);
            return data;
            
        } catch (error) {
            console.error('❌ Error collecting form data:', error);
            return null;
        }
    }
    
    async makePrediction(data) {
        try {
            console.log('🔮 Making prediction request...');
            
            const response = await fetch(`${this.API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('✅ Prediction result:', result);
            
            this.displayResults(result);
            
        } catch (error) {
            console.error('❌ Prediction error:', error);
            
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('Unable to connect to the prediction server. Please make sure the API is running on http://localhost:5000');
            } else {
                throw new Error(`Prediction failed: ${error.message}`);
            }
        }
    }
    
    displayResults(result) {
        const risk = result.risk;
        const confidence = result.confidence || 0;
        const probabilities = result.probabilities || {};
        
        // Update risk circle
        this.updateRiskCircle(risk, confidence, probabilities);
        
        // Update statistics
        this.updateRiskStats(result);
        
        // Update probability breakdown
        this.updateProbabilityBreakdown(probabilities);
        
        // Generate recommendations
        this.generateRecommendations(result, this.getFormValues());
        
        // Show results with animation
        this.showResults();
    }
    
    updateRiskCircle(risk, confidence, probabilities) {
        const riskCircle = document.getElementById('risk-circle');
        const riskPercentage = document.getElementById('risk-percentage-display');
        const riskLabel = document.getElementById('risk-label');
        
        if (!riskCircle || !riskPercentage || !riskLabel) return;
        
        const isHighRisk = risk === 1;
        const percentage = Math.round(confidence * 100);
        
        // Update text
        riskPercentage.textContent = `${percentage}%`;
        riskLabel.textContent = isHighRisk ? 'HIGH RISK' : 'LOW RISK';
        
        // Update styling
        riskCircle.className = `risk-circle ${isHighRisk ? 'high-risk' : 'low-risk'}`;
        
        // Animate the circle
        setTimeout(() => {
            const targetPercentage = isHighRisk ? 
                Math.round((probabilities.high_risk || probabilities.high_risk_probability || 0) * 100) :
                Math.round((probabilities.low_risk || probabilities.low_risk_probability || 0) * 100);
            
            this.animateRiskCircle(riskCircle, targetPercentage, isHighRisk);
        }, 500);
    }
    
    animateRiskCircle(circle, percentage, isHighRisk) {
        const color = isHighRisk ? '#ef4444' : '#10b981';
        const degrees = (percentage / 100) * 360;
        
        circle.style.background = `conic-gradient(
            from 0deg, 
            ${color} 0deg, 
            ${color} ${degrees}deg, 
            transparent ${degrees}deg
        )`;
    }
    
    updateRiskStats(result) {
        const confidenceValue = document.getElementById('confidence-value');
        const modelUsed = document.getElementById('model-used');
        
        if (confidenceValue) {
            confidenceValue.textContent = `${Math.round((result.confidence || 0) * 100)}%`;
        }
        
        if (modelUsed) {
            modelUsed.textContent = result.model_used || 'Random Forest';
        }
    }
    
    updateProbabilityBreakdown(probabilities) {
        const container = document.getElementById('probability-breakdown');
        if (!container) return;
        
        const lowRisk = probabilities.low_risk || probabilities.low_risk_probability || 0;
        const highRisk = probabilities.high_risk || probabilities.high_risk_probability || 0;
        
        container.innerHTML = `
            <div class="probability-item">
                <div class="probability-label">Low Risk</div>
                <div class="probability-bar">
                    <div class="probability-fill low-risk" style="width: ${lowRisk * 100}%"></div>
                </div>
                <div class="probability-value">${Math.round(lowRisk * 100)}%</div>
            </div>
            <div class="probability-item">
                <div class="probability-label">High Risk</div>
                <div class="probability-bar">
                    <div class="probability-fill high-risk" style="width: ${highRisk * 100}%"></div>
                </div>
                <div class="probability-value">${Math.round(highRisk * 100)}%</div>
            </div>
        `;
        
        // Animate the bars
        setTimeout(() => {
            container.querySelectorAll('.probability-fill').forEach(fill => {
                fill.style.transition = 'width 1s ease-out';
            });
        }, 100);
    }
    
    generateRecommendations(result, formData) {
        const container = document.getElementById('recommendations');
        if (!container) return;
        
        const recommendations = this.getPersonalizedRecommendations(result.risk, formData);
        
        container.innerHTML = `
            <h3><i class="fas fa-lightbulb"></i> Personalized Recommendations</h3>
            <div class="recommendation-list">
                ${recommendations.map(rec => `
                    <div class="recommendation-item">
                        <i class="fas ${rec.icon}"></i>
                        <div class="recommendation-text">${rec.text}</div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    getPersonalizedRecommendations(risk, formData) {
        const recommendations = [];
        
        // General recommendations
        if (risk === 1) {
            recommendations.push({
                icon: 'fa-user-md',
                text: 'Consult with a healthcare provider immediately for a comprehensive cardiac evaluation.'
            });
        }
        
        // Exercise recommendations
        const exerciseHours = parseFloat(formData['Exercise Hours Per Week']) || 0;
        if (exerciseHours < 2.5) {
            recommendations.push({
                icon: 'fa-running',
                text: 'Increase physical activity to at least 150 minutes of moderate exercise per week.'
            });
        }
        
        // Smoking recommendations
        if (formData['Smoking'] === 'Yes') {
            recommendations.push({
                icon: 'fa-smoking-ban',
                text: 'Consider quitting smoking - it\'s one of the most important steps for heart health.'
            });
        }
        
        // Stress recommendations
        const stressLevel = parseInt(formData['Stress Level']) || 5;
        if (stressLevel > 7) {
            recommendations.push({
                icon: 'fa-leaf',
                text: 'Practice stress management techniques like meditation, deep breathing, or yoga.'
            });
        }
        
        // Cholesterol recommendations
        const cholesterol = parseInt(formData['Cholesterol']) || 0;
        if (cholesterol > 240) {
            recommendations.push({
                icon: 'fa-apple-alt',
                text: 'Follow a heart-healthy diet low in saturated fats and cholesterol.'
            });
        }
        
        // Weight recommendations
        if (formData['Obesity'] === 'Yes') {
            recommendations.push({
                icon: 'fa-weight',
                text: 'Work with a healthcare provider on a safe weight management plan.'
            });
        }
        
        // Default recommendations if none specific
        if (recommendations.length === 0) {
            recommendations.push(
                {
                    icon: 'fa-heart',
                    text: 'Maintain regular check-ups with your healthcare provider.'
                },
                {
                    icon: 'fa-carrot',
                    text: 'Continue following a balanced, heart-healthy diet.'
                },
                {
                    icon: 'fa-dumbbell',
                    text: 'Keep up your current exercise routine and stay active.'
                }
            );
        }
        
        return recommendations;
    }
    
    getFormValues() {
        const formData = new FormData(this.form);
        const values = {};
        for (let [key, value] of formData.entries()) {
            values[key] = value;
        }
        return values;
    }
    
    showResults() {
        if (this.resultsContainer) {
            this.resultsContainer.classList.remove('hidden');
            
            // Smooth scroll to results
            setTimeout(() => {
                this.resultsContainer.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start' 
                });
            }, 100);
        }
    }
    
    hideResults() {
        if (this.resultsContainer) {
            this.resultsContainer.classList.add('hidden');
        }
    }
    
    showError(message) {
        const errorMessage = document.getElementById('error-message');
        if (errorMessage) {
            errorMessage.textContent = message;
        }
        
        if (this.errorContainer) {
            this.errorContainer.classList.remove('hidden');
        }
        
        console.error('❌ Error:', message);
    }
    
    hideError() {
        if (this.errorContainer) {
            this.errorContainer.classList.add('hidden');
        }
    }
    
    setLoadingState(loading) {
        if (!this.predictBtn) return;
        
        if (loading) {
            this.predictBtn.disabled = true;
            this.predictBtn.classList.add('loading');
            this.predictBtn.querySelector('.btn-text').innerHTML = `
                <i class="fas fa-spinner fa-spin"></i>
                Analyzing...
            `;
        } else {
            this.predictBtn.disabled = false;
            this.predictBtn.classList.remove('loading');
            this.predictBtn.querySelector('.btn-text').innerHTML = `
                <i class="fas fa-brain"></i>
                Analyze Risk
            `;
        }
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas ${type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
                <span>${message}</span>
            </div>
        `;
        
        // Add styles for notification
        const styles = `
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border-radius: 12px;
                padding: 16px 20px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
                z-index: 1000;
                animation: slideInRight 0.3s ease-out;
                max-width: 400px;
            }
            .notification-error {
                border-left: 4px solid #ef4444;
            }
            .notification-content {
                display: flex;
                align-items: center;
                gap: 12px;
                color: #1f2937;
            }
            .notification-error i {
                color: #ef4444;
            }
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        
        // Add styles to document if not already added
        if (!document.getElementById('notification-styles')) {
            const styleSheet = document.createElement('style');
            styleSheet.id = 'notification-styles';
            styleSheet.textContent = styles;
            document.head.appendChild(styleSheet);
        }
        
        document.body.appendChild(notification);
        
        // Remove notification after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideInRight 0.3s ease-out reverse';
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }
    
    async testAPIConnection() {
        try {
            const response = await fetch(`${this.API_BASE_URL}/`);
            if (response.ok) {
                console.log('✅ API connection successful');
            } else {
                console.warn('⚠️ API responded but may have issues');
            }
        } catch (error) {
            console.warn('⚠️ API connection failed:', error.message);
            console.log('💡 Make sure to run: python api.py');
        }
    }
}

// Global functions for HTML event handlers
function resetForm() {
    if (app && app.form) {
        app.form.reset();
        app.currentStep = 1;
        app.updateStepDisplay();
        app.updateProgress();
        app.updateNavigationButtons();
        app.hideResults();
        app.hideError();
        app.initializeStressSlider();
        
        // Clear all validation feedback
        document.querySelectorAll('.input-feedback').forEach(feedback => {
            feedback.textContent = '';
            feedback.className = 'input-feedback';
        });
        
        document.querySelectorAll('input, select').forEach(field => {
            field.style.borderColor = '';
        });
        
        app.showNotification('Form reset successfully', 'info');
    }
}

function hideError() {
    if (app) {
        app.hideError();
    }
}

function downloadReport() {
    // Generate and download a simple report
    const results = {
        timestamp: new Date().toISOString(),
        risk: document.getElementById('risk-label')?.textContent || 'Unknown',
        confidence: document.getElementById('confidence-value')?.textContent || 'Unknown',
        model: document.getElementById('model-used')?.textContent || 'Unknown'
    };
    
    const reportContent = `
Heart Attack Risk Assessment Report
=====================================

Generated: ${new Date().toLocaleString()}

Risk Level: ${results.risk}
Confidence: ${results.confidence}
Model Used: ${results.model}

Disclaimer: This prediction is for informational purposes only 
and should not replace professional medical advice.
    `;
    
    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `heart_risk_report_${new Date().getTime()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    if (app) {
        app.showNotification('Report downloaded successfully!', 'info');
    }
}

// Initialize the app
const app = new HeartRiskApp();