from django.contrib.auth.views import LoginView
from django.contrib import messages
from django.contrib.auth import login
from django.shortcuts import redirect
from django.views.generic.edit import FormView
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

        super().form_valid(form)

        authorization_url = (
            "/oauth2/authorize/"
            f"?client_id={client_id}"
            f"&response_type=code"
            f"&redirect_uri={redirect_uri}"
        )
        return redirect(authorization_url)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        client_id = self.request.GET.get('client_id')
        context['client_id'] = client_id
        redirect_uri = self.request.GET.get('redirect_uri')
        context['redirect_uri'] = redirect_uri
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

        # Save the new user
        user = form.save()

        # Log in the new user
        login(self.request, user)

        authorization_url = (
            "/oauth2/authorize/"
            f"?client_id={client_id}"
            f"&response_type=code"
            f"&redirect_uri={redirect_uri}"
        )
        return redirect(authorization_url)

    def form_invalid(self, form):
        # messages.error(self.request, 'There was an error with your signup. Please try again.')
        return super().form_invalid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        client_id = self.request.GET.get('client_id')
        context['client_id'] = client_id
        redirect_uri = self.request.GET.get('redirect_uri')
        context['redirect_uri'] = redirect_uri
        return context
