FROM alpine
MAINTAINER Jeremy Jess <jeremytjess@gmail.com>

# application folder
COPY . /app
WORKDIR /app

RUN apk add --no-cache py3-pip uwsgi uwsgi-python3 \
	&& pip3 install --upgrade pip \
	&& pip3 install -r requirements.txt

# expose web server port
# only http, for ssl use reverse proxy
EXPOSE 80

# exectute start up script
ENTRYPOINT ["python3"]

CMD ["routes.py"]