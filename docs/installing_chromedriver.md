# Installing Chrome and Chromedriver

## Installing Chrome

Update your packages.

```bash
sudo apt update
```

Download and install chrome

```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
```

Then you can install chrome from the downloaded file.

```bash
sudo dpkg -i google-chrome-stable_current_amd64.deb
sudo apt-get install -f
```

Check Chrome is installed correctly.

```bash
google-chrome --version
```

This version is important, you will need it to get the chromedriver.

## Installing Chromedriver

You can download the chromedriver from this location [Download Chromedriver](https://chromedriver.chromium.org/downloads).
But you need the correct version, so remember which version of chrome you have from the steps above and download the correct chromedriver.

Download the chromedriver, making sure you replace the link with your link to match your version of chrome.

```bash
wget https://chromedriver.storage.googleapis.com/108.0.5359.71/chromedriver_linux64.zip
```

You will get a zip file, you need to unzip it:

```bash
unzip chromedriver_linux64.zip
```

You then need to move the file to the correct location, so you can find when you need it.

```bash
sudo mv chromedriver /usr/bin/chromedriver
sudo chown root:root /usr/bin/chromedriver
sudo chmod +x /usr/bin/chromedriver
```

## Test Installation

Run the command

```bash
chromedriver --url-base=/wd/hub
```

You should see something like this:

```
Starting ChromeDriver 108.0.5359.71 (1e0e3868ee06e91ad636a874420e3ca3ae3756ac-refs/branch-heads/5359@{#1016}) on port 9515
Only local connections are allowed.
Please see https://chromedriver.chromium.org/security-considerations for suggestions on keeping ChromeDriver safe.
ChromeDriver was started successfully.
```
