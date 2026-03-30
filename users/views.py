from django.contrib.auth.views import LoginView
from django.contrib import messages
from django.contrib.auth import login, logout
from django.shortcuts import redirect
from django.views.generic.edit import FormView
from django.http import JsonResponse
from django.urls import reverse


from .forms import CustomSignUpForm


class CustomLoginView(LoginView):
    def form_invalid(self, form):
        messages.error(
            self.request, 'Invalid username or password. Please try again.')
        return super().form_invalid(form)

    def form_valid(self, form):
        client_id = self.request.GET.get('client_id')
        if not client_id:
            messages.error(self.request, 'Client ID is missing.')
            return super().form_invalid(form)
        
        redirect_uri = self.request.GET.get('redirect_uri')
        if not redirect_uri:
            messages.error(self.request, 'redirect_uri is missing.')
            return super().form_invalid(form)
        
        code_challenge = self.request.GET.get('code_challenge')
        if not code_challenge:
            messages.error(self.request, 'code_challenge is missing.')
            return super().form_invalid(form)

        code_challenge_method = self.request.GET.get('code_challenge_method')
        if not code_challenge_method:
            messages.error(self.request, 'code_challenge_method is missing.')
            return super().form_invalid(form)


        super().form_valid(form)

        authorization_url = (
            reverse('oauth2_provider:authorize') + 
            f"?client_id={client_id}"
            f"&response_type=code"
            f"&redirect_uri={redirect_uri}"
            f"&code_challenge={code_challenge}"
            f"&code_challenge_method={code_challenge_method}"
        )
        return redirect(authorization_url)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['client_id'] = self.request.GET.get('client_id')
        context['redirect_uri'] = self.request.GET.get('redirect_uri')
        context['code_challenge'] = self.request.GET.get('code_challenge')
        context['code_challenge_method'] = self.request.GET.get('code_challenge_method')
        return context


class CustomSignUpView(FormView):
    template_name = 'registration/signup.html'
    form_class = CustomSignUpForm

    def form_valid(self, form):
        # Extract client_id from the query parameters
        client_id = self.request.GET.get('client_id')
        if not client_id:
            messages.error(self.request, 'Client ID is missing.')
            return redirect('signup')
        
        redirect_uri = self.request.GET.get('redirect_uri')
        if not redirect_uri:
            messages.error(self.request, 'redirect_uri is missing.')
            return redirect('signup')
        
        code_challenge = self.request.GET.get('code_challenge')
        if not code_challenge:
            messages.error(self.request, 'code_challenge is missing.')
            return super().form_invalid(form)

        code_challenge_method = self.request.GET.get('code_challenge_method')
        if not code_challenge_method:
            messages.error(self.request, 'code_challenge_method is missing.')
            return super().form_invalid(form)

        # Save the new user
        user = form.save()

        # Log in the new user
        login(self.request, user)

        authorization_url = (
            reverse('oauth2_provider:authorize') + 
            f"?client_id={client_id}"
            f"&response_type=code"
            f"&redirect_uri={redirect_uri}"
            f"&code_challenge={code_challenge}"
            f"&code_challenge_method={code_challenge_method}"
        )
        return redirect(authorization_url)

    def form_invalid(self, form):
        # messages.error(self.request, 'There was an error with your signup. Please try again.')
        return super().form_invalid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['client_id'] = self.request.GET.get('client_id')
        context['redirect_uri'] = self.request.GET.get('redirect_uri')
        context['code_challenge'] = self.request.GET.get('code_challenge')
        context['code_challenge_method'] = self.request.GET.get('code_challenge_method')
        return context
    
# removes session from database
def logout_view(request):
    logout(request)
    return JsonResponse({"status": "success"})